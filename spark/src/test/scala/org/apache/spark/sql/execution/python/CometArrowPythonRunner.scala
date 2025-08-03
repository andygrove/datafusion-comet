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

package org.apache.spark.sql.execution.python

import java.io.DataInputStream
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.python.{BasePythonRunner, ChainedPythonFunctions}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

class CometArrowPythonRunner(
    funcs: Seq[ChainedPythonFunctions],
    evalType: Int,
    argOffsets: Array[Array[Int]],
//                             protected override val schema: StructType,
//                             protected override val timeZoneId: String,
//                             protected override val largeVarTypes: Boolean,
//                             protected override val workerConf: Map[String, String],
    val pythonMetrics: Map[String, SQLMetric],
    jobArtifactUUID: Option[String])
    extends BasePythonRunner[Iterator[ColumnarBatch], ColumnarBatch](
      funcs,
      evalType,
      argOffsets,
      jobArtifactUUID) {

  override protected def newWriterThread(
      env: SparkEnv,
      worker: Socket,
      inputIterator: Iterator[Iterator[ColumnarBatch]],
      partitionIndex: Int,
      context: TaskContext): WriterThread = {
    throw new UnsupportedOperationException()
  }

  override protected def newReaderIterator(
      stream: DataInputStream,
      writerThread: WriterThread,
      startTime: Long,
      env: SparkEnv,
      worker: Socket,
      pid: Option[Int],
      releasedOrClosed: AtomicBoolean,
      context: TaskContext): Iterator[ColumnarBatch] = {
    throw new UnsupportedOperationException()
  }
}
