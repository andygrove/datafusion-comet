/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.comet

import java.util.concurrent.atomic.AtomicBoolean
import java.util.function.LongSupplier

import scala.util.control.NonFatal

import org.apache.arrow.memory.{AllocationListener, AllocationOutcome, OutOfMemoryException}
import org.apache.spark.internal.Logging
import org.apache.spark.memory.{MemoryConsumer, MemoryMode, SparkOutOfMemoryError, TaskMemoryManager}

import org.apache.comet.CometConf

/**
 * Charges one task's JVM-side Arrow allocations to Spark's off-heap execution pool, and refuses
 * an allocation the pool cannot cover.
 *
 * `CometArrowAllocator` is a process-wide `RootAllocator` with no limit, but the bytes a task's
 * allocator hands out are resident in the container all the same. Charged here, they appear in
 * `TaskMemoryManager.showMemoryUsage`, are arbitrated against Spark's other off-heap consumers,
 * and are bounded by the same pool.
 *
 * '''Reserved before it is allocated.''' Arrow calls [[onPreAllocation]] before it allocates a
 * buffer, and documents it as the one callback that may throw to terminate the allocation. The
 * reservation is taken there, in whole blocks, and if Spark cannot cover the allocation whatever
 * it did grant is handed straight back and the allocation is refused with Arrow's own
 * `OutOfMemoryException`, the exception Arrow's vectors already clean up after. That is
 * `try_grow` rather than `grow`: nothing is allocated that is not covered, and there is no
 * overcommit. Nothing can spill these buffers either, since the JVM holds them until the batch is
 * done with, so a refusal fails the allocating task, the way a Spark consumer fails when it
 * cannot acquire memory.
 *
 * '''Ownership.''' One instance is created per task and attached to that task's Arrow allocator
 * by [[CometTaskArrowAllocator]]. Arrow calls the listener of the allocator that makes a buffer
 * on allocation, and the listener of the allocator that '''owns''' it once the last reference
 * drops, on whichever thread that happens to be. `AllocationListener` is handed nothing but a
 * size, so binding the listener to an allocator is the only way to attribute a release. Reading
 * `TaskContext` inside the callbacks would get it wrong: a buffer exported over the C Data
 * Interface is released by whichever native thread drops it last, and that thread has no task
 * context installed.
 *
 * '''What is charged is what the allocator owns, not a running total of the callbacks.''' The two
 * come apart because ownership moves between allocators without either listener hearing about it.
 * When two allocators hold references to one buffer and the owner lets go first, Arrow hands the
 * charge to the other through `transferBalance`, which calls no listener, and the final
 * `onRelease` goes to the new owner. That is the everyday path rather than a corner:
 * `ColumnarBatchArrowReader` retains every batch it streams to native into its own allocator, a
 * child of the unaccounted root, and then closes the source. Summing the callbacks would charge
 * the task for each such batch until the task ended. So the callbacks only say when to look, and
 * the size comes from the task allocator's accountant, which Arrow keeps right across transfers.
 * Every callback trims the reservation to what is owned and pending, so a transfer out is
 * reflected at the next allocation or release on the task's allocator, and the reader calls
 * [[reconcile]] as soon as it closes a source, so that a task stops paying for a batch once
 * native has it. A transfer in is charged at the next allocation, which is refused if the pool
 * cannot cover both.
 *
 * '''Only [[onPreAllocation]] may throw.''' Arrow documents that the other callbacks cannot, and
 * `BaseAllocator.buffer` marks the allocation successful before calling `onAllocation`, so
 * throwing from any of them loses a buffer Arrow has already created. They only ever settle or
 * trim the reservation, and contain anything the memory manager throws. How the acquisition in
 * [[onPreAllocation]] can fail is described at [[acquire]].
 *
 * '''Lock order.''' This listener's monitor is taken before Spark's and never the other way
 * round: [[onPreAllocation]] holds ours across `acquireExecutionMemory`, and [[acquire]]
 * additionally takes the `TaskMemoryManager` monitor itself, so that the two usage snapshots
 * either side of that call cannot be split by another consumer. [[getUsed]] and [[spill]] must
 * therefore stay lock-free, because Spark calls both while holding its own monitor: were either
 * to take ours, a native reservation arriving through `CometTaskMemoryManager` on a Comet Tokio
 * thread could hold Spark's monitor and wait for ours while an Arrow allocation on the same task
 * held ours and waited for Spark's. For the same reason, nothing reachable from a `spill`
 * callback may allocate JVM Arrow memory.
 */
private[comet] class CometArrowAllocationListener(taskMemoryManager: TaskMemoryManager)
    extends MemoryConsumer(taskMemoryManager, 0L, MemoryMode.OFF_HEAP)
    with AllocationListener {

  import CometArrowAllocationListener._

  /**
   * Bytes the task's allocator currently owns, read from Arrow's own accountant, which is an
   * atomic, so [[getUsed]] can read it without taking this listener's monitor; see the lock order
   * note above. Set by [[bind]], because the allocator is built with this listener and so cannot
   * exist before it.
   */
  @volatile private var owned: LongSupplier = NothingOwned

  /** Bytes currently reserved with Spark. Guarded by this listener's monitor. */
  private var reserved = 0L

  /**
   * Bytes admitted by [[onPreAllocation]] whose allocation Arrow has not reported yet. They count
   * against the reservation, so that two allocations on different threads cannot both be admitted
   * against the same headroom. Guarded by this listener's monitor.
   */
  private var pending = 0L

  /** Set once the owning task has finished. Volatile so [[getUsed]] can read it lock-free. */
  @volatile private var completed = false

  /** Attaches the listener to what it accounts for. Called once, before the allocator is used. */
  private[comet] def bind(owned: LongSupplier): Unit = this.owned = owned

  /**
   * Reserves what the allocation needs before Arrow makes it, or refuses it. Once the task has
   * finished there is nothing left to charge, so a straggling allocation goes through.
   */
  override def onPreAllocation(size: Long): Unit = synchronized {
    if (!completed) {
      val needed = owned.getAsLong + pending + size
      if (reserved < needed) {
        reserved += acquireAtLeast(needed - reserved, roundUpToBlock(needed) - reserved, size)
      }
      pending += size
    }
  }

  override def onAllocation(size: Long): Unit = settle(size)

  /** Arrow's own limit refused the allocation after it was admitted here, so hand it back. */
  override def onFailedAllocation(size: Long, outcome: AllocationOutcome): Boolean = {
    settle(size)
    false
  }

  override def onRelease(size: Long): Unit = reconcile()

  /** Trims the reservation after buffers moved out of the allocator without a callback. */
  private[comet] def reconcile(): Unit = settle(0L)

  /**
   * Reports what the task's allocator owns. Spark reads this for spill-victim ordering,
   * `showMemoryUsage` and end-of-task leak reporting. The inherited `used` counter stays at zero
   * because this consumer never calls `acquireMemory` or `allocatePage`; the reservation is made
   * on Arrow's behalf, not through the consumer's own page accounting.
   *
   * Reports zero once the task has finished, so that buffers deliberately allowed to outlive
   * their task are not reported by `cleanUpAllAllocatedMemory` as a Spark memory leak.
   */
  override def getUsed: Long = if (completed) 0L else owned.getAsLong

  /** The JVM holds these buffers until the batch is done with, so there is nothing to spill. */
  override def spill(size: Long, trigger: MemoryConsumer): Long = 0L

  /**
   * Drops the whole reservation and stops accounting.
   *
   * Called from the owning task's completion listener. Anything still alive afterwards is a
   * buffer that outlives its task, which the process-wide allocator exists to allow; those
   * releases are ignored rather than charged to whichever task happens to be running by then.
   */
  private[comet] def taskCompleted(): Unit = {
    try {
      synchronized {
        completed = true
        releaseDownTo(0L)
      }
    } catch {
      case NonFatal(e) => warnOnReleaseFailure(e)
    }
  }

  /** Bytes currently reserved with Spark on this task's behalf. Visible for testing. */
  private[comet] def reservedBytes: Long = synchronized(reserved)

  /**
   * Moves `size` bytes out of pending, since Arrow has put them in the allocator's accountant by
   * the time it calls back, and trims the reservation to what is owned and pending, rounded up to
   * a block. The rounding keeps a change within a block from reaching Spark, although a change
   * that crosses a block boundary still does every time. Releasing cannot fail in practice, but
   * this is reached from callbacks that may not throw, so anything unforeseen is contained.
   */
  private def settle(size: Long): Unit = {
    try {
      synchronized {
        pending = math.max(0L, pending - size)
        if (!completed) {
          releaseDownTo(roundUpToBlock(owned.getAsLong + pending))
        }
      }
    } catch {
      case NonFatal(e) => warnOnReleaseFailure(e)
    }
  }

  /**
   * Returns everything reserved above `keep`, in one call rather than one per block, because
   * `releaseExecutionMemory` synchronizes on the executor-wide pool. Called with this listener's
   * monitor held.
   */
  private def releaseDownTo(keep: Long): Unit = {
    if (reserved > keep) {
      taskMemoryManager.releaseExecutionMemory(reserved - keep, this)
      reserved = keep
    }
  }

  /**
   * Asks Spark for `request` bytes, which takes the reservation up to a block boundary so that
   * the next small allocation does not go straight back into Spark's lock, and returns what it
   * granted. A grant short of `request` is accepted as long as it covers `shortfall`, because the
   * rounding is there to save lock traffic and must not be the reason an allocation that fits is
   * refused. Anything less is handed back and the allocation refused.
   */
  private def acquireAtLeast(shortfall: Long, request: Long, size: Long): Long = {
    val granted = acquire(request, size)
    if (granted < shortfall) {
      if (granted > 0L) {
        taskMemoryManager.releaseExecutionMemory(granted, this)
      }
      throw refusal(size, shortfall, granted, null)
    }
    granted
  }

  /**
   * Asks Spark for `request` bytes and returns what it granted, or refuses the allocation if the
   * acquisition fails.
   *
   * It is fallible in three ways, and only the first is caught by `NonFatal`: it runs other
   * consumers' `spill`, which turns a task interrupt into a `RuntimeException` and an I/O failure
   * into a `SparkOutOfMemoryError`, and the execution pool itself parks in `lock.wait()`, so
   * killing a task can raise a plain `InterruptedException`. Each becomes the refusal's cause,
   * and an interrupt also re-arms the thread's flag.
   *
   * `acquireExecutionMemory` takes its first grant from the pool and only then asks other
   * consumers to spill, so when a spill throws it has already charged the task for bytes it never
   * reports back, and nothing would release them before `cleanUpAllAllocatedMemory` at the very
   * end of the task. They are measured as the change in what the pool says this task holds, and
   * released before the refusal is thrown.
   *
   * Spark reports that figure per task rather than per consumer, so it only measures '''our'''
   * grant if nothing else in the task can move it while we are looking. Both snapshots and the
   * acquisition therefore run as one transaction under the `TaskMemoryManager` monitor. That is
   * the same monitor `acquireExecutionMemory` takes and holds for its whole duration, spills
   * included, and it is reentrant, so taking it here only widens that window to cover the two
   * reads. Every acquisition in the task funnels through that method, so with it held no other
   * consumer can take memory between a snapshot and the call and have it released here.
   *
   * What the monitor does not cover is a release, which reaches the pool without it. Another
   * consumer returning memory, or a spill that frees some bytes before throwing, makes the figure
   * too small, which is the safe direction: too small leaves bytes charged to the task until it
   * ends, which is what would have happened anyway. `request` bounds it from above.
   */
  private def acquire(request: Long, size: Long): Long = taskMemoryManager.synchronized {
    val heldBefore = taskMemoryManager.getMemoryConsumptionForThisTask
    try {
      taskMemoryManager.acquireExecutionMemory(request, this)
    } catch {
      case e @ (_: InterruptedException | _: SparkOutOfMemoryError | NonFatal(_)) =>
        if (e.isInstanceOf[InterruptedException]) {
          // Refusing is the only way out of an Arrow callback, but the cancellation must not be
          // lost with it: re-arm the flag so the task still observes it.
          Thread.currentThread().interrupt()
        }
        val orphaned = math.max(
          0L,
          math.min(taskMemoryManager.getMemoryConsumptionForThisTask - heldBefore, request))
        if (orphaned > 0L) {
          taskMemoryManager.releaseExecutionMemory(orphaned, this)
        }
        throw refusal(size, request, 0L, e)
    }
  }

  private def refusal(
      size: Long,
      needed: Long,
      granted: Long,
      cause: Throwable): OutOfMemoryException = {
    // As Spark's own consumers do before failing, so the task's consumers are in the log.
    taskMemoryManager.showMemoryUsage()
    new OutOfMemoryException(
      s"Unable to reserve $needed bytes of Spark off-heap execution memory for a JVM Arrow " +
        s"allocation of $size bytes, got $granted. Increase spark.memory.offHeap.size, or set " +
        s"${CometConf.COMET_LEGACY_UNBOUNDED_JVM_ARROW_MEMORY.key}=true to restore the previous " +
        "behavior of not charging JVM Arrow allocations to Spark.",
      cause)
  }
}

object CometArrowAllocationListener extends Logging {

  /**
   * Batching granularity for reservations. Arrow allocates per buffer and
   * `acquireExecutionMemory` takes an executor-wide lock, so the reservation is grown and shrunk
   * in whole blocks and only block-crossing changes reach Spark. Deliberately not configurable:
   * it trades lock chatter against reservation slack and has no plausible per-workload tuning.
   */
  private[comet] val BLOCK_SIZE = 1024L * 1024L

  private val NothingOwned: LongSupplier = () => 0L

  private val releaseFailureLogged = new AtomicBoolean(false)

  private def roundUpToBlock(bytes: Long): Long =
    ((bytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE

  private def warnOnReleaseFailure(e: Throwable): Unit = {
    if (releaseFailureLogged.compareAndSet(false, true)) {
      logWarning(
        "Failed to return JVM Arrow memory to Spark's memory manager. The task keeps it " +
          "reserved until it ends, when Spark releases everything the task still holds.",
        e)
    }
  }
}
