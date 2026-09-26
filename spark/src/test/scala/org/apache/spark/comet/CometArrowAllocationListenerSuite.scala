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

import java.io.{InterruptedIOException, IOException}
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}

import scala.jdk.CollectionConverters._

import org.scalatest.funsuite.AnyFunSuite

import org.apache.arrow.c.Data
import org.apache.arrow.memory.OutOfMemoryException
import org.apache.arrow.vector.{FieldVector, IntVector, VectorSchemaRoot}
import org.apache.spark.{CometTaskMemoryManager, SparkConf, SparkContext, TaskContext, TaskContextImpl}
import org.apache.spark.memory.{MemoryConsumer, MemoryManager, MemoryMode, SparkOutOfMemoryError, TaskMemoryManager}
import org.apache.spark.sql.comet.execution.arrow.CometArrowStream
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.ThreadUtils

import org.apache.comet.{CometArrowAllocator, CometArrowImportAllocator, CometConf}
import org.apache.comet.vector.NativeUtil

/**
 * Tests that JVM Arrow allocations are charged to Spark, that they are charged to the task that
 * made them rather than whichever task happens to be on the releasing thread, that what is
 * charged follows Arrow's ownership as buffers move between allocators, and that an allocation
 * Spark cannot cover is refused before Arrow makes it, leaving no buffer, reservation or grant
 * behind.
 */
class CometArrowAllocationListenerSuite extends AnyFunSuite {

  private val blockSize = CometArrowAllocationListener.BLOCK_SIZE
  private val poolBytes = 64L * 1024 * 1024

  // ---------------------------------------------------------------------------------------------
  // Reservation arithmetic. Driven through the listener directly, with a stand-in for the
  // allocator's accountant, since Arrow's rounding policy would otherwise decide the sizes under
  // test.
  // ---------------------------------------------------------------------------------------------

  test("allocations are charged to the current task in whole blocks") {
    withTask() { task =>
      val owner = new StubOwner(task.taskMemoryManager)
      val listener = owner.listener

      // Far smaller than a block, so the reservation should round up to exactly one block.
      owner.allocate(128L)
      assert(listener.reservedBytes == blockSize)

      // Still inside the first block, so Spark is not asked again.
      owner.allocate(1024L)
      assert(listener.reservedBytes == blockSize)

      listener.taskCompleted()
    }
  }

  test("a request larger than a block rounds up to a block multiple") {
    withTask() { task =>
      val owner = new StubOwner(task.taskMemoryManager)
      owner.allocate(blockSize * 3 + 7L)
      val listener = owner.listener
      // Rounded up rather than sized to the exact deficit, so growth leaves headroom and the next
      // small allocation does not go straight back into Spark.
      assert(listener.reservedBytes == blockSize * 4)
      listener.taskCompleted()
    }
  }

  test("releasing returns whole blocks to Spark") {
    withTask() { task =>
      val owner = new StubOwner(task.taskMemoryManager)
      val listener = owner.listener
      owner.allocate(blockSize * 2)
      assert(listener.reservedBytes == blockSize * 2)

      owner.release(blockSize * 2)
      assert(listener.reservedBytes == 0L)
      listener.taskCompleted()
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Which allocator is handed out, and what it charges.
  // ---------------------------------------------------------------------------------------------

  test("a real Arrow allocation from the task allocator is charged to that task") {
    withTask() { task =>
      val allocator = CometTaskArrowAllocator.forCurrentTask()
      assert(allocator ne CometArrowAllocator)
      val buf = allocator.buffer(blockSize)
      try {
        assert(reservedFor(task) == blockSize)
        assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == blockSize)
      } finally {
        buf.close()
      }
      assert(reservedFor(task) == 0L)
    }
  }

  test("a child of the task allocator is charged to the same task") {
    withTask() { task =>
      // The Python runner and the native Arrow source cut their own children. Arrow passes the
      // parent's listener down, so they are accounted without knowing anything about it.
      val child =
        CometTaskArrowAllocator.forCurrentTask().newChildAllocator("probe", 0L, Long.MaxValue)
      try {
        val buf = child.buffer(blockSize)
        try {
          assert(reservedFor(task) == blockSize)
        } finally {
          buf.close()
        }
      } finally {
        child.close()
      }
      assert(reservedFor(task) == 0L)
    }
  }

  test("the allocators on either side of the FFI boundary are not accounted") {
    withTask() { task =>
      // Establish the task allocator first, so this asserts "not charged" rather than "no task".
      CometTaskArrowAllocator.forCurrentTask()
      // Buffers allocated for export come from the listener-less root, and imports from a
      // listener-less child of it, because native accounts for what it retains. Arrow reports an
      // imported buffer to the importing allocator's listener at full capacity, so giving the
      // import allocator a listener would charge Spark for native memory.
      for (allocator <- Seq(CometArrowAllocator, CometArrowImportAllocator)) {
        val buf = allocator.buffer(blockSize)
        try {
          assert(reservedFor(task) == 0L, s"${allocator.getName} was charged to the task")
        } finally {
          buf.close()
        }
      }
    }
  }

  test("no active task uses the unaccounted root allocator") {
    TaskContext.unset()
    // Broadcast coalescing on the driver, and native threads pulling a stream, have no task.
    assert(CometTaskArrowAllocator.forCurrentTask() eq CometArrowAllocator)
  }

  test("on-heap mode uses the unaccounted root allocator") {
    withTask(offHeap = false) { task =>
      // Comet's on-heap mode exists so the Spark SQL suite can run without off-heap memory.
      // Charging an off-heap consumer there would be wrong.
      assert(CometTaskArrowAllocator.forCurrentTask() eq CometArrowAllocator)
      assert(CometTaskArrowAllocator.listenerForTask(task.taskAttemptId).isEmpty)
    }
  }

  test("the legacy setting hands tasks the unaccounted root, and is off by default") {
    // Read from the executors' SparkConf, so each arm runs a task in a context of its own.
    def taskGetsRoot(legacy: Option[String]): Boolean = {
      val conf = new SparkConf()
        .setMaster("local[1]")
        .setAppName(getClass.getSimpleName)
        .set("spark.memory.offHeap.enabled", "true")
        .set("spark.memory.offHeap.size", poolBytes.toString)
      legacy.foreach(conf.set(CometConf.COMET_LEGACY_UNBOUNDED_JVM_ARROW_MEMORY.key, _))
      val sc = new SparkContext(conf)
      try {
        sc.parallelize(Seq(0), 1)
          .map(_ => CometTaskArrowAllocator.forCurrentTask() eq CometArrowAllocator)
          .collect()
          .head
      } finally {
        sc.stop()
      }
    }
    assert(!taskGetsRoot(None), "a task was not accounted by default")
    assert(taskGetsRoot(Some("true")), "the legacy setting did not restore the unaccounted root")
  }

  // ---------------------------------------------------------------------------------------------
  // Ownership: a release is attributed to the task that allocated, not to the releasing thread.
  // ---------------------------------------------------------------------------------------------

  test("a release on a thread with no task context is charged to the allocating task") {
    withTask() { task =>
      val buf = CometTaskArrowAllocator.forCurrentTask().buffer(blockSize)
      assert(reservedFor(task) == blockSize)

      // This is what happens when a shuffle-read batch is handed on to a native operator: native
      // pins it and drops it later from a Tokio worker with no task context installed. Reading
      // TaskContext in onRelease would ignore this release and leave the task charged for memory
      // it had already freed.
      ThreadUtils.runInNewThread("detached-release")(buf.close())

      assert(reservedFor(task) == 0L)
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == 0L)
    }
  }

  test("a release under a different task does not touch that task's accounting") {
    withTask() { taskA =>
      val bufA = CometTaskArrowAllocator.forCurrentTask().buffer(blockSize)
      withTask() { taskB =>
        val bufB = CometTaskArrowAllocator.forCurrentTask().buffer(blockSize)
        assert(reservedFor(taskB) == blockSize)

        // A's buffer, released while B's context is on the thread. B must not pay for it.
        bufA.close()

        assert(reservedFor(taskB) == blockSize)
        assert(reservedFor(taskA) == 0L)
        bufB.close()
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // What is charged follows ownership. Arrow moves a buffer's charge between allocators with
  // `transferBalance`, which calls no listener, so a tally of the callbacks drifts from what the
  // task's allocator actually owns.
  // ---------------------------------------------------------------------------------------------

  test("a buffer whose ownership moves to another allocator stops being charged to the task") {
    withTask() { task =>
      val stream = CometArrowAllocator.newChildAllocator("stream", 0L, Long.MaxValue)
      try {
        val allocator = CometTaskArrowAllocator.forCurrentTask()
        val source = allocator.buffer(blockSize * 4)
        assert(reservedFor(task) == blockSize * 4)

        // What ColumnarBatchArrowReader does to every batch it streams to native: retain the
        // buffers into the stream's allocator, then close the source. Arrow makes the stream's
        // allocator the owner, and neither listener is told.
        val retained = source.getReferenceManager.retain(source, stream)
        try {
          source.close()
          assert(allocator.getAllocatedMemory == 0L)
          assert(listenerFor(task).getUsed == 0L)
        } finally {
          // Native drops it later, and the release goes to the stream allocator's listener, which
          // is the root's no-op. Nothing on the task's allocator hears about it.
          retained.close()
        }

        // With no reconcile, the reservation catches up at the next callback on the task's
        // allocator: one block for the new buffer, not four for a buffer the task no longer owns.
        val next = allocator.buffer(128L)
        assert(reservedFor(task) == blockSize)
        next.close()
        assert(reservedFor(task) == 0L)
      } finally {
        stream.close()
      }
    }
  }

  test("batches streamed to native stop being charged to the task that read them") {
    withTask() { task =>
      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val importer = CometArrowAllocator.newChildAllocator("importer", 0L, Long.MaxValue)
      val numBatches = 16
      // Half a block of values per batch, which Arrow allocates as one block-sized buffer.
      val rowsPerBatch = (blockSize / 2 / IntVector.TYPE_WIDTH).toInt
      val batches = Iterator.tabulate(numBatches) { b =>
        val vector = new IntVector("i", allocator)
        vector.allocateNew(rowsPerBatch)
        (0 until rowsPerBatch).foreach(i => vector.set(i, b + i))
        vector.setValueCount(rowsPerBatch)
        val root = new VectorSchemaRoot(java.util.Arrays.asList[FieldVector](vector))
        root.setRowCount(rowsPerBatch)
        NativeUtil.rootAsBatch(root): ColumnarBatch
      }
      try {
        // The same export a native operator's JVM input takes, drained here by a JVM importer
        // standing in for native's ScanExec.
        val stream = CometArrowStream.fromColumnarBatchIter(
          batches,
          StructType(Seq(StructField("i", IntegerType))),
          CometArrowStream.NATIVE_TIMEZONE,
          "listener-suite")
        val reader = Data.importArrayStream(importer, stream)
        var imported = 0
        try {
          while (reader.loadNextBatch()) {
            imported += 1
            // Native holds this batch now, and the task no longer owns any of it, so it should not
            // be paying for it either. Summing the callbacks would have kept a block for every
            // batch read so far, and waiting for the task's next allocation would have kept one
            // for this batch, which for a broadcast coalesced into one batch is all of it.
            assert(allocator.getAllocatedMemory == 0L)
            assert(
              reservedFor(task) == 0L,
              s"reserved ${reservedFor(task)} bytes after handing over $imported batches")
          }
        } finally {
          reader.close()
        }
        assert(imported == numBatches)
      } finally {
        importer.close()
      }
    }
  }

  test("a buffer adopted from another allocator is charged while the task owns it") {
    withTask() { task =>
      val other = CometArrowAllocator.newChildAllocator("other", 0L, Long.MaxValue)
      try {
        val allocator = CometTaskArrowAllocator.forCurrentTask()
        val foreign = other.buffer(blockSize)
        // The mirror image: the task allocator takes a reference and the original owner lets go,
        // so Arrow makes the task allocator the owner without telling either listener.
        val adopted = foreign.getReferenceManager.retain(foreign, allocator)
        try {
          foreign.close()
          assert(allocator.getAllocatedMemory == blockSize)
          assert(listenerFor(task).getUsed == blockSize)
        } finally {
          // The final release then reaches this task's listener for bytes it never saw allocated.
          adopted.close()
        }
        // That must not leave the task's figure short, so the next allocation is charged in full.
        val buf = allocator.buffer(blockSize)
        try {
          assert(reservedFor(task) == blockSize)
        } finally {
          buf.close()
        }
        assert(reservedFor(task) == 0L)
      } finally {
        other.close()
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Task completion, and buffers that outlive their task.
  // ---------------------------------------------------------------------------------------------

  test("task completion releases the whole reservation and closes the allocator") {
    val allocatorName = withTask() { task =>
      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val buf = allocator.buffer(blockSize)
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == blockSize)
      buf.close()

      task.context.markTaskCompleted(None)

      assert(CometTaskArrowAllocator.listenerForTask(task.taskAttemptId).isEmpty)
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == 0L)
      allocator.getName
    }
    assert(!rootChildNames().contains(allocatorName))
  }

  test("a buffer outliving its task parks the allocator until it is released") {
    withTask() { task =>
      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val buf = allocator.buffer(blockSize)

      task.context.markTaskCompleted(None)

      // The reservation goes back to Spark even though the buffer is still alive: the task is
      // over, and leaving it charged would be reported as a Spark memory leak.
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == 0L)
      assert(CometTaskArrowAllocator.listenerForTask(task.taskAttemptId).isEmpty)
      // Arrow treats closing an allocator that still owns bytes as a leak, so it has to stay open.
      assert(rootChildNames().contains(allocator.getName))

      // A late release is ignored rather than charged to whoever is running by then...
      buf.close()
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == 0L)

      // ...and the drained allocator is reaped by the next task, so the root does not accumulate
      // one child per task attempt.
      withTask() { _ => CometTaskArrowAllocator.forCurrentTask() }
      assert(!rootChildNames().contains(allocator.getName))
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Refusal. An allocation Spark cannot cover is refused in onPreAllocation, before Arrow has made
  // it, and nothing is left behind: no buffer, no reservation, and no grant charged to the task.
  // ---------------------------------------------------------------------------------------------

  test("an allocation the pool cannot cover is refused, and one that fits still succeeds") {
    // Two blocks of budget, one of them held by a consumer that cannot give anything back.
    withTask(pool = blockSize * 2) { task =>
      val other = new OtherConsumer(task.taskMemoryManager)
      assert(other.take(blockSize) == blockSize)

      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val refused = intercept[OutOfMemoryException](allocator.buffer(blockSize * 2))
      assert(refused.getMessage.contains(CometConf.COMET_LEGACY_UNBOUNDED_JVM_ARROW_MEMORY.key))
      // Spark granted the block it had before coming up short. That partial grant is handed back
      // rather than kept, and Arrow never allocated anything.
      assert(allocator.getAllocatedMemory == 0L)
      assert(reservedFor(task) == 0L)
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == blockSize)

      val buf = allocator.buffer(blockSize)
      try {
        assert(reservedFor(task) == blockSize)
      } finally {
        buf.close()
      }
    }
  }

  test("a grant short of a block is accepted when it covers the allocation") {
    // Reservations are requested in whole blocks to save lock traffic, but a pool with less than a
    // block left must still admit an allocation that fits in what is left.
    val left = 64L * 1024
    withTask(pool = blockSize) { task =>
      val other = new OtherConsumer(task.taskMemoryManager)
      assert(other.take(blockSize - left) == blockSize - left)

      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val buf = allocator.buffer(1024L)
      try {
        assert(reservedFor(task) == left)
        // What does not fit in what is left is still refused.
        intercept[OutOfMemoryException](allocator.buffer(left))
        assert(reservedFor(task) == left)
      } finally {
        buf.close()
      }
      // A reservation that is not a whole block is returned once nothing needs it.
      assert(reservedFor(task) == 0L)
    }
  }

  test("an allocation still in flight counts against the reservation") {
    withTask(pool = blockSize) { task =>
      val owner = new StubOwner(task.taskMemoryManager)
      val listener = owner.listener

      // Admitted, but Arrow has not reported it yet...
      listener.onPreAllocation(blockSize)
      assert(listener.reservedBytes == blockSize)
      // ...so a second allocation racing it on another thread cannot be admitted against the same
      // block.
      intercept[OutOfMemoryException](listener.onPreAllocation(blockSize))
      assert(listener.reservedBytes == blockSize)

      owner.made(blockSize)
      owner.release(blockSize)
      assert(listener.reservedBytes == 0L)
      listener.taskCompleted()
    }
  }

  for ((label, failure) <- Seq(
      "an I/O failure" -> new IOException("spill failed"),
      "an interrupted spill" -> new InterruptedIOException("task killed"))) {
    test(s"$label while acquiring refuses the allocation and leaks nothing") {
      // Exactly one block of budget, already taken by a consumer whose spill throws, so the
      // acquisition has to go through Spark's spill path and comes back throwing.
      withTask(pool = blockSize) { task =>
        val hostile = new OtherConsumer(task.taskMemoryManager, Some(failure))
        assert(hostile.take(blockSize) == blockSize)

        val allocator = CometTaskArrowAllocator.forCurrentTask()
        val refused = intercept[OutOfMemoryException](allocator.buffer(blockSize))
        // The memory manager's failure is the cause, which is what says the acquisition really
        // went down the spill path rather than being refused for want of memory alone.
        assert(refused.getCause != null)
        assert(allocator.getAllocatedMemory == 0L)
        assert(reservedFor(task) == 0L)
        assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == blockSize)
      }
    }
  }

  test("an interrupt while acquiring refuses the allocation and leaves the flag set") {
    // Spark's execution pool parks in `lock.wait()` when a task is below its fair share, so a task
    // kill raises a plain InterruptedException out of `acquireExecutionMemory`. TestMemoryManager
    // never parks, so the interrupt is injected through a failing spill instead.
    withTask(pool = blockSize) { task =>
      val hostile =
        new OtherConsumer(task.taskMemoryManager, Some(new InterruptedException("task killed")))
      assert(hostile.take(blockSize) == blockSize)

      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val refused = intercept[OutOfMemoryException](allocator.buffer(blockSize))
      assert(refused.getCause.isInstanceOf[InterruptedException])
      assert(allocator.getAllocatedMemory == 0L)
      assert(reservedFor(task) == 0L)
      // Cleared here as well as asserted, so the flag does not leak into the next test.
      assert(Thread.interrupted(), "the interrupt was swallowed instead of being re-armed")
    }
  }

  test("a partial grant lost to a failing spill is released rather than stranded") {
    // One block already taken, one still in the pool, and a two-block request: Spark hands over the
    // block it has and only then asks the other consumer to spill, which throws. It never reports
    // the block it already took, so nothing would release it before the task ended.
    withTask(pool = blockSize * 2) { task =>
      val hostile =
        new OtherConsumer(task.taskMemoryManager, Some(new IOException("spill failed")))
      assert(hostile.take(blockSize) == blockSize)

      val allocator = CometTaskArrowAllocator.forCurrentTask()
      intercept[OutOfMemoryException](allocator.buffer(blockSize * 2))
      assert(reservedFor(task) == 0L)
      // Only the other consumer's block is left. Without releasing the orphan this would be two
      // blocks, one of them charged to the task and owned by nobody.
      assert(task.taskMemoryManager.getMemoryConsumptionForThisTask == blockSize)
    }
  }

  test("a concurrent acquisition is not released as this listener's lost grant") {
    // The orphan is measured as a change in the task's total consumption, which Spark reports per
    // task rather than per consumer. If another consumer could acquire between the snapshot taken
    // before the acquisition and the acquisition itself, its bytes would be counted as the orphan
    // and released here, leaving that consumer holding bytes the pool no longer charges for. Both
    // snapshots and the call therefore run as one transaction under the
    // TaskMemoryManager monitor. This forces the interleaving that transaction exists to exclude.
    val spare = 1024L
    withTask(pool = blockSize * 2 + spare) { task =>
      val hostile =
        new OtherConsumer(task.taskMemoryManager, Some(new IOException("spill failed")))
      assert(hostile.take(blockSize) == blockSize)

      val interloper = new OtherConsumer(task.taskMemoryManager)
      // Once the acquisition has taken what was left, the pool is empty and Spark answers the
      // interloper by asking the hostile consumer to spill, which throws. That is a legitimate
      // outcome for the interloper and not what is under test here; what it ends up holding is.
      val interloperThread = daemonThread("interloper") {
        try interloper.take(spare)
        catch { case _: SparkOutOfMemoryError => }
      }

      // Fires once, in place of the snapshot taken before the acquisition: exactly the window the
      // transaction has to close. The interloper either runs to completion here, which is the bug,
      // or blocks on the monitor the acquisition is holding, which is the fix.
      task.snapshotHook.set(() => {
        interloperThread.start()
        awaitBlockedOrFinished(interloperThread)
      })

      val allocator = CometTaskArrowAllocator.forCurrentTask()
      intercept[OutOfMemoryException](allocator.buffer(blockSize * 2))
      interloperThread.join(30000L)
      assert(
        !interloperThread.isAlive,
        "the interloper never finished; the transaction deadlocked")
      // Every byte the task is charged for belongs to someone. Without the transaction the
      // listener counts the interloper's bytes as its own orphan and releases them, and the task
      // comes out charged for less than the other two consumers hold.
      assert(
        listenerFor(task).reservedBytes + hostile.getUsed + interloper.getUsed ==
          task.taskMemoryManager.getMemoryConsumptionForThisTask,
        "the listener released bytes belonging to another consumer")
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Lock order. Spark calls getUsed and spill while holding the TaskMemoryManager monitor, and the
  // listener holds its own monitor while waiting for that one.
  // ---------------------------------------------------------------------------------------------

  test("the usage snapshot does not take the reservation monitor") {
    withTask() { task =>
      val buf = CometTaskArrowAllocator.forCurrentTask().buffer(blockSize)
      try {
        val listener = listenerFor(task)
        val used = new AtomicLong(-1L)
        val spilled = new AtomicLong(-1L)
        listener.synchronized {
          // A native reservation arriving through CometTaskMemoryManager on a Tokio thread holds
          // Spark's monitor here. If either call waited on this one, it would deadlock against an
          // Arrow allocation on the same task that already holds this monitor and wants Spark's.
          val probe = daemonThread("lock-order-probe") {
            used.set(listener.getUsed)
            spilled.set(listener.spill(blockSize, listener))
          }
          probe.start()
          probe.join(30000L)
          assert(!probe.isAlive, "getUsed or spill blocked on the reservation monitor")
        }
        assert(used.get == blockSize)
        assert(spilled.get == 0L)
      } finally {
        buf.close()
      }
    }
  }

  test("concurrent Arrow and native reservations make progress") {
    // Two blocks of budget, and the native side asks for both, so whenever the Arrow side holds
    // one the native grant comes up short and Spark walks its consumer list, calling getUsed and
    // spill on the Arrow listener while holding its own monitor. Either side can come up short,
    // and an Arrow allocation refused for it is a legitimate outcome: finishing is what is tested.
    withTask(pool = blockSize * 2) { task =>
      val allocator = CometTaskArrowAllocator.forCurrentTask()
      val native = new CometTaskMemoryManager(0L, task.taskAttemptId)
      val failure = new AtomicReference[Throwable]()

      val arrowThread = loopingThread("arrow-allocations", failure) {
        try allocator.buffer(blockSize).close()
        catch { case _: OutOfMemoryException => }
      }
      val nativeThread = loopingThread("native-reservations", failure) {
        native.releaseMemory(native.acquireMemory(blockSize * 2))
      }

      Seq(arrowThread, nativeThread).foreach(_.start())
      Seq(arrowThread, nativeThread).foreach { t =>
        t.join(60000L)
        assert(!t.isAlive, s"${t.getName} did not finish; concurrent reservations deadlocked")
      }

      Option(failure.get).foreach(e => fail("a worker failed", e))
      assert(allocator.getAllocatedMemory == 0L)
      assert(native.getUsed == 0L)
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Fixtures.
  // ---------------------------------------------------------------------------------------------

  /**
   * A listener with a stand-in for the task allocator's accountant, calling the listener the way
   * Arrow would.
   */
  private class StubOwner(tmm: TaskMemoryManager) {
    val listener = new CometArrowAllocationListener(tmm)
    private val owned = new AtomicLong(0L)
    listener.bind(() => owned.get)

    /** What `BaseAllocator.buffer` does around a successful allocation. */
    def allocate(bytes: Long): Unit = {
      listener.onPreAllocation(bytes)
      made(bytes)
    }

    /** The second half of an allocation that `onPreAllocation` has already admitted. */
    def made(bytes: Long): Unit = {
      owned.addAndGet(bytes)
      listener.onAllocation(bytes)
    }

    def release(bytes: Long): Unit = {
      owned.addAndGet(-bytes)
      listener.onRelease(bytes)
    }
  }

  /**
   * Any other consumer in the task: holds memory and cannot give it back. With a `spillFailure`,
   * its spill throws that instead, so `trySpillAndAcquire` throws on its behalf.
   */
  private class OtherConsumer(tmm: TaskMemoryManager, spillFailure: Option[Exception] = None)
      extends MemoryConsumer(tmm, 0L, MemoryMode.OFF_HEAP) {
    def take(bytes: Long): Long = acquireMemory(bytes)
    override def spill(size: Long, trigger: MemoryConsumer): Long = spillFailure match {
      case Some(failure) => throw failure
      case None => 0L
    }
  }

  /**
   * Runs one action immediately after the usage snapshot the listener takes before asking Spark
   * for memory, so a test can drive what happens in the window between that snapshot and the
   * acquisition. The real value is read first, which is what makes the window the one under test:
   * running the action before the read would fold whatever it does into the snapshot itself.
   */
  private class HookedTaskMemoryManager(
      memoryManager: MemoryManager,
      taskAttemptId: Long,
      val hook: AtomicReference[Runnable] = new AtomicReference[Runnable]())
      extends TaskMemoryManager(memoryManager, taskAttemptId) {
    override def getMemoryConsumptionForThisTask(): Long = {
      val held = super.getMemoryConsumptionForThisTask()
      val pending = hook.getAndSet(null)
      if (pending != null) pending.run()
      held
    }
  }

  private case class TaskFixture(
      context: TaskContextImpl,
      taskMemoryManager: HookedTaskMemoryManager) {
    def taskAttemptId: Long = context.taskAttemptId
    def snapshotHook: AtomicReference[Runnable] = taskMemoryManager.hook
  }

  private def reservedFor(task: TaskFixture): Long =
    CometTaskArrowAllocator.listenerForTask(task.taskAttemptId).map(_.reservedBytes).getOrElse(0L)

  private def listenerFor(task: TaskFixture): CometArrowAllocationListener =
    CometTaskArrowAllocator.listenerForTask(task.taskAttemptId).get

  /** Runs the body in a task of its own, completed afterwards, and returns what the body does. */
  private def withTask[T](offHeap: Boolean = true, pool: Long = poolBytes)(
      f: TaskFixture => T): T = {
    val (context, taskMemoryManager) =
      TestTasks.newTask(pool, offHeap)(new HookedTaskMemoryManager(_, _))
    val task = TaskFixture(context, taskMemoryManager)
    TestTasks.withInstalled(context)(f(task))
  }

  /** An unstarted daemon thread, so a test can choose the moment it runs. */
  private def daemonThread(name: String)(body: => Unit): Thread = {
    val thread = new Thread(() => body)
    thread.setDaemon(true)
    thread.setName(name)
    thread
  }

  /**
   * Waits until the thread is either blocked on a monitor or finished, whichever happens first,
   * so that a test can tell the two orderings apart without depending on timing.
   */
  private def awaitBlockedOrFinished(thread: Thread): Unit = {
    val deadline = System.currentTimeMillis() + 30000L
    var state = thread.getState
    while (state != Thread.State.BLOCKED && state != Thread.State.TERMINATED &&
      System.currentTimeMillis() < deadline) {
      Thread.sleep(1L)
      state = thread.getState
    }
  }

  /** Runs the body 500 times on an unstarted daemon thread, recording the first failure. */
  private def loopingThread(name: String, failure: AtomicReference[Throwable])(
      body: => Unit): Thread = daemonThread(name) {
    try {
      var i = 0
      while (i < 500) {
        body
        i += 1
      }
    } catch {
      case t: Throwable => failure.compareAndSet(null, t)
    }
  }

  private def rootChildNames(): Set[String] =
    CometArrowAllocator.getChildAllocators.asScala.map(_.getName).toSet
}
