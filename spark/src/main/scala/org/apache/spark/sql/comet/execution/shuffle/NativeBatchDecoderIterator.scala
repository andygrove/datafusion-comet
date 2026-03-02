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

package org.apache.spark.sql.comet.execution.shuffle

import java.io.InputStream
import java.nio.channels.Channels

import org.apache.spark.TaskContext
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.apache.comet.{CometConf, Native}
import org.apache.comet.vector.NativeUtil

/**
 * This iterator reads an Arrow IPC stream from a Spark shuffle input stream using a stateful
 * native StreamReader. Each partition's data is a single Arrow IPC stream with body compression.
 */
case class NativeBatchDecoderIterator(
    in: InputStream,
    taskContext: TaskContext,
    decodeTime: SQLMetric)
    extends Iterator[ColumnarBatch] {

  private var isClosed = false
  private val native = new Native()
  private val nativeUtil = new NativeUtil()
  private val tracingEnabled = CometConf.COMET_TRACING_ENABLED.get()
  private var currentBatch: ColumnarBatch = null

  if (taskContext != null) {
    taskContext.addTaskCompletionListener[Unit](_ => {
      close()
    })
  }

  // Stream Arrow IPC data incrementally via a ReadableByteChannel backed by
  // a BufferedInputStream. Native code reads from this channel via JNI callbacks,
  // so decoding can start as soon as the first IPC message arrives.
  private val (readerHandle, fieldCount) = {
    if (in == null) {
      (0L, 0)
    } else {
      val buffered =
        new java.io.BufferedInputStream(in, NativeBatchDecoderIterator.READ_BUFFER_SIZE)
      val channel = Channels.newChannel(buffered)
      val handle = native.createShuffleStreamReader(channel, tracingEnabled)
      if (handle == 0L) {
        (0L, 0)
      } else {
        val cols = native.getShuffleStreamReaderColumns(handle)
        (handle, cols)
      }
    }
  }

  private var batch = fetchNext()

  def hasNext(): Boolean = {
    if (readerHandle == 0 || isClosed) {
      return false
    }
    if (batch.isDefined) {
      return true
    }

    // Release the previous batch.
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }

    batch = fetchNext()
    if (batch.isEmpty) {
      close()
      return false
    }
    true
  }

  def next(): ColumnarBatch = {
    if (!hasNext) {
      throw new NoSuchElementException
    }

    val nextBatch = batch.get

    currentBatch = nextBatch
    batch = None
    currentBatch
  }

  private def fetchNext(): Option[ColumnarBatch] = {
    if (readerHandle == 0 || isClosed) {
      return None
    }

    val startTime = System.nanoTime()
    val result = nativeUtil.getNextBatch(
      fieldCount,
      (arrayAddrs, schemaAddrs) => {
        native.nextShuffleBatch(readerHandle, arrayAddrs, schemaAddrs, tracingEnabled)
      })
    decodeTime.add(System.nanoTime() - startTime)

    result
  }

  def close(): Unit = {
    synchronized {
      if (!isClosed) {
        if (currentBatch != null) {
          currentBatch.close()
          currentBatch = null
        }
        if (readerHandle != 0) {
          native.closeShuffleStreamReader(readerHandle)
        }
        if (in != null) {
          in.close()
        }
        nativeUtil.close()
        isClosed = true
      }
    }
  }
}

object NativeBatchDecoderIterator {
  private val READ_BUFFER_SIZE = 64 * 1024
}
