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

import java.io.{DataInputStream, DataOutputStream}
import java.net.Socket
import java.nio.channels.Channels
import java.util.concurrent.atomic.AtomicBoolean

import scala.collection.JavaConverters._

import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.api.python.{BasePythonRunner, ChainedPythonFunctions}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.comet.execution.arrow.ArrowReaderIterator
import org.apache.spark.sql.comet.util.Utils
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
    extends BasePythonRunner[ColumnarBatch, ColumnarBatch](
      funcs,
      evalType,
      argOffsets,
      jobArtifactUUID) {

  override protected def newWriterThread(
      env: SparkEnv,
      worker: Socket,
      inputIterator: Iterator[ColumnarBatch],
      partitionIndex: Int,
      context: TaskContext): WriterThread = {

    new CometArrowWriterThread(env, worker, inputIterator, partitionIndex, context)
  }

  /**
   * WriterThread that writes ColumnarBatch data as Arrow streams to Python worker processes.
   */
  private class CometArrowWriterThread(
      env: SparkEnv,
      worker: Socket,
      inputIterator: Iterator[ColumnarBatch],
      partitionIndex: Int,
      context: TaskContext)
      extends WriterThread(env, worker, inputIterator, partitionIndex, context)
      with Logging {

    override protected def writeCommand(dataOut: DataOutputStream): Unit = {
      // Write the command and function information as expected by Python worker
      // This follows the same pattern as BasePythonRunner for sending UDF metadata
      dataOut.writeInt(evalType)
      dataOut.writeInt(funcs.length)
      funcs.zipWithIndex.foreach { case (chainedFunc, index) =>
        dataOut.writeInt(chainedFunc.funcs.length)
        chainedFunc.funcs.foreach { func =>
          dataOut.writeUTF(new String(func.command.toArray))
          dataOut.writeUTF(func.envVars.asScala.mkString(","))
          dataOut.writeUTF(func.pythonIncludes.asScala.mkString(","))
          dataOut.writeUTF(func.pythonExec)
          dataOut.writeUTF(func.pythonVer)
          dataOut.writeUTF(func.broadcastVars.asScala.mkString(","))
          dataOut.writeUTF(func.accumulator.toString)
        }
        dataOut.writeInt(argOffsets(index).length)
        argOffsets(index).foreach(dataOut.writeInt)
      }
    }

    override protected def writeIteratorToStream(dataOut: DataOutputStream): Unit = {
      while (writeNextInputToStream(dataOut)) {
        // Continue writing batches
      }
    }

    protected def writeNextInputToStream(dataOut: DataOutputStream): Boolean = {
      if (inputIterator.hasNext) {
        val batch = inputIterator.next()
        if (batch.numRows() > 0) {
          writeArrowBatch(batch, dataOut)
        }
        true
      } else {
        false
      }
    }

    private def writeArrowBatch(batch: ColumnarBatch, dataOut: DataOutputStream): Unit = {
      try {
        // Use utility method from common module to write Arrow batch
        Utils.writeArrowBatchToStream(batch, dataOut)
      } catch {
        case e: Exception =>
          logError("Error writing Arrow batch to Python worker", e)
          throw e
      }
    }

    def close(): Unit = {
      try {
        // Resources are managed by the utility method
        logDebug("Closing CometArrowWriterThread")
      } catch {
        case e: Exception =>
          logError("Error closing CometArrowWriterThread", e)
      }
    }
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

    // Create an ArrowReaderIterator to read Arrow IPC format from the Python worker
    val arrowReader = new ArrowReaderIterator(
      Channels.newChannel(stream),
      s"Python worker ${pid.getOrElse("unknown")}")

    // Wrap the ArrowReaderIterator to handle exceptions and thread coordination
    new Iterator[ColumnarBatch] {
      private var finished = false

      override def hasNext: Boolean = {
        if (finished) {
          return false
        }

        // Check for exceptions from the writer thread
        if (writerThread != null && writerThread.exception.isDefined) {
          throw writerThread.exception.get
        }

        // Check if the reader has more data
        if (!arrowReader.hasNext) {
          finished = true
          false
        } else {
          true
        }
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) {
          throw new NoSuchElementException("End of stream")
        }

        try {
          arrowReader.next()
        } catch {
          case e: Exception =>
            finished = true
            throw e
        }
      }
    }
  }
}
