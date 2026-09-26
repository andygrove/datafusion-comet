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

import java.util.Properties
import java.util.concurrent.atomic.AtomicLong

import org.apache.spark.{SparkConf, TaskContext, TaskContextImpl}
import org.apache.spark.executor.TaskMetrics
import org.apache.spark.memory.{MemoryManager, TaskMemoryManager, TestMemoryManager}

/**
 * Spark tasks built by hand, each over a memory pool of its own, for driving the JVM Arrow
 * accounting without a `SparkContext`.
 */
private[comet] object TestTasks {

  /** Task attempt ids are keys in a process-wide map, so no two tasks may share one. */
  private val nextTaskAttemptId = new AtomicLong(1000L)

  /**
   * A task over a pool of `pool` bytes, off-heap unless `offHeap` is false, whose
   * `TaskMemoryManager` comes from `newTaskMemoryManager` so that a caller can subclass it.
   */
  def newTask[M <: TaskMemoryManager](pool: Long, offHeap: Boolean = true)(
      newTaskMemoryManager: (MemoryManager, Long) => M): (TaskContextImpl, M) = {
    val conf = new SparkConf(false)
    if (offHeap) {
      conf
        .set("spark.memory.offHeap.enabled", "true")
        .set("spark.memory.offHeap.size", pool.toString)
    }
    val memoryManager = new TestMemoryManager(conf)
    memoryManager.limit(pool)
    val taskAttemptId = nextTaskAttemptId.getAndIncrement()
    val taskMemoryManager = newTaskMemoryManager(memoryManager, taskAttemptId)
    val context = new TaskContextImpl(
      stageId = 0,
      stageAttemptNumber = 0,
      partitionId = 0,
      numPartitions = 1,
      taskAttemptId = taskAttemptId,
      attemptNumber = 0,
      taskMemoryManager = taskMemoryManager,
      localProperties = new Properties,
      metricsSystem = null,
      taskMetrics = TaskMetrics.empty,
      cpus = 1,
      resources = Map.empty)
    (context, taskMemoryManager)
  }

  /**
   * Runs the body with the task installed on this thread, then completes the task, which fires
   * the completion listener that drops the Arrow reservation and is harmless if it already ran,
   * and restores whatever was installed before.
   */
  def withInstalled[T](context: TaskContextImpl)(f: => T): T = {
    val previous = TaskContext.get()
    TaskContext.setTaskContext(context)
    try {
      f
    } finally {
      try {
        context.markTaskCompleted(None)
        context.taskMemoryManager.cleanUpAllAllocatedMemory()
      } finally {
        if (previous == null) TaskContext.unset() else TaskContext.setTaskContext(previous)
      }
    }
  }
}
