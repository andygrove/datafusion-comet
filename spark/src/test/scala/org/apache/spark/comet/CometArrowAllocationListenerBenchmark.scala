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

import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong}

import org.apache.arrow.memory.{ArrowBuf, BufferAllocator, RootAllocator}
import org.apache.spark.{CometTaskMemoryManager, TaskContext}
import org.apache.spark.benchmark.{Benchmark, BenchmarkBase}
import org.apache.spark.memory.{MemoryConsumer, MemoryManager, TaskMemoryManager}

/**
 * Measures what charging JVM Arrow allocations to Spark costs, since it is on by default.
 *
 * Each case runs the same allocate/release loop twice: once against a plain `RootAllocator`,
 * which is what Comet did before [[CometArrowAllocationListener]] existed and what
 * `spark.comet.legacy.unboundedJvmArrowMemory=true` restores, and once against a task allocator
 * carrying the listener. Reservation call counts are printed under each table, because the
 * interesting variable is not the per-buffer bookkeeping but how often a buffer size crosses a
 * block boundary and has to go into `TaskMemoryManager` at all.
 *
 * To run this benchmark:
 * {{{
 * SPARK_GENERATE_BENCHMARK_FILES=1 make benchmark-org.apache.spark.comet.CometArrowAllocationListenerBenchmark
 * }}}
 */
object CometArrowAllocationListenerBenchmark extends BenchmarkBase {

  private val blockSize = CometArrowAllocationListener.BLOCK_SIZE
  private val poolBytes = 1024L * 1024L * 1024L

  override def runBenchmarkSuite(mainArgs: Array[String]): Unit = {
    runBenchmark("JVM Arrow allocations charged to Spark") {
      // Many sub-block buffers, the shape a codegen output vector's validity and offset buffers
      // take. None of them comes close to a block, so the listener should never reach Spark after
      // the first one.
      allocateAndRelease("small buffers", bufferSize = 128L, buffersPerIteration = 512)
      // A wide batch: many medium buffers alive at once, one block boundary crossed per few
      // buffers on the way up and the same on the way down.
      allocateAndRelease(
        "wide batch buffers",
        bufferSize = 64L * 1024L,
        buffersPerIteration = 128)
      // Worst case for the block batching: every allocation crosses a boundary, so every one of
      // them takes the executor-wide lock in `acquireExecutionMemory`.
      allocateAndRelease("block-sized buffers", bufferSize = blockSize, buffersPerIteration = 8)
      // Same, with a second thread reserving from the same pool the way Comet's native side does.
      // The pool has room for the whole set plus that thread's block, because a smaller one would
      // refuse the allocations rather than slow them down, so what this measures is contention for
      // the pool's lock rather than for its capacity. In the unaccounted arm only the pressure
      // thread touches the pool: the delta is what the Arrow side adds by competing.
      allocateAndRelease(
        "block-sized buffers under native pressure",
        bufferSize = blockSize,
        buffersPerIteration = 8,
        pool = blockSize * 9,
        nativePressure = true)
      // Per call site rather than per buffer, but it is the cost the root allocator `val` did not
      // have: a TaskContext lookup and a concurrent map read.
      allocatorLookup()
    }
  }

  private def allocateAndRelease(
      name: String,
      bufferSize: Long,
      buffersPerIteration: Int,
      pool: Long = poolBytes,
      nativePressure: Boolean = false): Unit = {
    val benchmark =
      new Benchmark(
        s"$name (${buffersPerIteration}x$bufferSize)",
        buffersPerIteration,
        output = output)

    // Both allocators are built once, outside the timed body, so what is measured is the
    // steady-state cost of allocating and releasing rather than the cost of standing a task up.
    val root = new RootAllocator(Long.MaxValue)
    try {
      withTaskAllocator(pool) { (accounted, memory) =>
        withNativePressure(nativePressure) {
          benchmark.addCase("not accounted") { _ =>
            churn(root, bufferSize, buffersPerIteration)
          }
          benchmark.addCase("accounted") { _ =>
            churn(accounted, bufferSize, buffersPerIteration)
          }
          benchmark.run()

          // One more round with the counters zeroed, to report how often a single iteration
          // reaches the memory manager. That, rather than the per-buffer bookkeeping, is the cost
          // that scales with buffer size.
          memory.reset()
          churn(accounted, bufferSize, buffersPerIteration)
          writeLine(s"  accounted: ${memory.summary(buffersPerIteration)} per iteration")
        }
      }
    } finally {
      root.close()
    }
  }

  private def allocatorLookup(): Unit = {
    val lookupsPerIteration = 100000
    val benchmark =
      new Benchmark("allocator lookup", lookupsPerIteration.toLong, output = output)

    // The "before" shape: call sites read a package-object `val`, which the JIT folds away
    // entirely. The interesting number is therefore the absolute cost of the second case.
    benchmark.addCase("process-wide val") { _ =>
      var i = 0
      var sink = 0
      while (i < lookupsPerIteration) {
        sink += System.identityHashCode(org.apache.comet.CometArrowAllocator)
        i += 1
      }
      assert(sink != Int.MinValue)
    }
    benchmark.addCase("forCurrentTask()") { _ =>
      withTaskAllocator() { (_, _) =>
        var i = 0
        var sink = 0
        while (i < lookupsPerIteration) {
          sink += System.identityHashCode(CometTaskArrowAllocator.forCurrentTask())
          i += 1
        }
        assert(sink != Int.MinValue)
      }
    }

    benchmark.run()
  }

  /**
   * Allocates the whole set, then releases it, so the peak is what the reservation has to cover.
   */
  private def churn(allocator: BufferAllocator, bufferSize: Long, count: Int): Unit = {
    val buffers = new Array[ArrowBuf](count)
    var i = 0
    while (i < count) {
      buffers(i) = allocator.buffer(bufferSize)
      i += 1
    }
    i = 0
    while (i < count) {
      buffers(i).close()
      i += 1
    }
  }

  /**
   * Runs the body against a task allocator, torn down afterwards, so that repeated iterations do
   * not accumulate reservations or allocators.
   */
  private def withTaskAllocator[T](pool: Long = poolBytes)(
      f: (BufferAllocator, CountingTaskMemoryManager) => T): T = {
    val (context, memory) = TestTasks.newTask(pool)(new CountingTaskMemoryManager(_, _))
    TestTasks.withInstalled(context)(f(CometTaskArrowAllocator.forCurrentTask(), memory))
  }

  /**
   * Hammers the task's pool from another thread for the duration of the body, when asked to, the
   * way native reservations do. Built on the task's thread, which `CometTaskMemoryManager` needs.
   */
  private def withNativePressure[T](enabled: Boolean)(f: => T): T = {
    if (!enabled) {
      f
    } else {
      val stop = new AtomicBoolean(false)
      val native = new CometTaskMemoryManager(0L, TaskContext.get().taskAttemptId())
      val thread = new Thread(() => {
        while (!stop.get()) {
          native.releaseMemory(native.acquireMemory(blockSize))
        }
      })
      thread.setDaemon(true)
      thread.setName("native-reservations")
      thread.start()
      try f
      finally {
        stop.set(true)
        thread.join()
      }
    }
  }

  private def writeLine(line: String): Unit = {
    // scalastyle:off println
    println(line)
    // scalastyle:on println
    output.foreach(_.write(s"$line\n".getBytes("UTF-8")))
  }

  private class CountingTaskMemoryManager(memoryManager: MemoryManager, taskAttemptId: Long)
      extends TaskMemoryManager(memoryManager, taskAttemptId) {
    private val acquires = new AtomicLong(0L)
    private val releases = new AtomicLong(0L)

    // Only the Arrow listener's own calls are counted, so the pressure thread's traffic does not
    // land in the reported figure.
    override def acquireExecutionMemory(required: Long, consumer: MemoryConsumer): Long = {
      if (consumer.isInstanceOf[CometArrowAllocationListener]) acquires.incrementAndGet()
      super.acquireExecutionMemory(required, consumer)
    }

    override def releaseExecutionMemory(size: Long, consumer: MemoryConsumer): Unit = {
      if (consumer.isInstanceOf[CometArrowAllocationListener]) releases.incrementAndGet()
      super.releaseExecutionMemory(size, consumer)
    }

    def reset(): Unit = {
      acquires.set(0L)
      releases.set(0L)
    }

    def summary(buffers: Int): String =
      s"${acquires.get()} acquire and ${releases.get()} release calls for $buffers buffers"
  }
}
