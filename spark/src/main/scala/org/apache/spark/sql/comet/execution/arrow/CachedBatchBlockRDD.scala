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

package org.apache.spark.sql.comet.execution.arrow

import java.io.InputStream
import java.nio.ByteBuffer

import org.apache.spark.{Partition, SparkException, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.columnar.CachedBatch
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.apache.comet.{CometBlockSource, CometShuffleBlockIterator}

/**
 * The cached batches of a Comet cache scan, read by a native `ShuffleScan` that decodes them.
 *
 * It holds the scan's batches after pruning, and hands each payload to native code as it is
 * stored, so the heap-to-off-heap copy of the selected buffers, their decompression and the Arrow
 * Java vectors that `CachedBatchIpc.Projection.load` builds all go: the native side decompresses
 * just the columns it projects with arrow-rs. The batches have no other reader, so there is no
 * `compute`.
 */
private[comet] class CachedBatchBlockRDD(prev: RDD[CachedBatch], numOutputRows: SQLMetric)
    extends RDD[ColumnarBatch](prev)
    with CometBlockSource {

  override protected def getPartitions: Array[Partition] = prev.partitions

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] =
    throw new UnsupportedOperationException(
      "Cached batches for a native cache scan are only read through a native ShuffleScan")

  override def computeAsShuffleBlockIterator(
      split: Partition,
      context: TaskContext): CometShuffleBlockIterator =
    new CachedBatchBlockIterator(prev.iterator(split, context), numOutputRows)
}

/**
 * Hands Comet cached batch payloads to a native `ShuffleScan`, one per `hasNext`, in the same
 * protocol that shuffle blocks use: the payload is copied into a direct buffer that native code
 * reads, and that stays valid until the next `hasNext`.
 *
 * The payload is held in heap chunks, which native code cannot address, so one copy remains. It
 * copies the whole payload rather than the selected columns' buffers, which the native decoder
 * picks out and decompresses itself.
 */
private class CachedBatchBlockIterator(batches: Iterator[CachedBatch], numOutputRows: SQLMetric)
    extends CometShuffleBlockIterator(InputStream.nullInputStream()) {

  private var buffer: ByteBuffer = _
  private var length = 0

  override def hasNext(): Int = {
    if (!batches.hasNext) {
      return -1
    }
    batches.next() match {
      case cached: CometCachedBatch =>
        val size = cached.bytes.size
        if (size > Int.MaxValue) {
          throw new SparkException(
            s"Comet cached batch of $size bytes is too large for the native cache scan. Set " +
              "spark.comet.exec.inMemoryCache.nativeDecode.enabled=false to read it on the JVM.")
        }
        buffer = CometShuffleBlockIterator.ensureCapacity(buffer, size.toInt)
        buffer.clear()
        cached.bytes.getChunks().foreach(chunk => buffer.put(chunk))
        length = size.toInt
        length
      case other =>
        throw new IllegalStateException(
          s"Unsupported cached batch type ${other.getClass.getName}")
    }
  }

  override def getBuffer: ByteBuffer = buffer

  override def getCurrentBlockLength: Int = length

  // A cached payload has no remote source to invalidate, and Comet wrote it in this application.
  override def requiresValidation(): Boolean = false

  override def onDecodeFailure(message: String): Unit = ()

  override def incRecordsRead(records: Long): Unit = numOutputRows += records
}
