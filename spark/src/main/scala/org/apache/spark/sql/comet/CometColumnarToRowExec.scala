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

package org.apache.spark.sql.comet

import scala.collection.JavaConverters._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{ColumnarToRowTransition, SparkPlan}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.util.Utils

import org.apache.comet.vector.CometVector

/**
 * This is currently an identical copy of Spark's ColumnarToRowExec except for removing the
 * code-gen features.
 *
 * This is moved into the Comet repo as the first step towards refactoring this to make the
 * interactions with CometVector more efficient to avoid some JNI overhead.
 */
case class CometColumnarToRowExec(child: SparkPlan)
    extends CometExec
    with ColumnarToRowTransition {
  // supportsColumnar requires to be only called on driver side, see also SPARK-37779.
  assert(Utils.isInRunningSparkTask || child.supportsColumnar)

  override def supportsColumnar: Boolean = false

  override def originalPlan: SparkPlan = child

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"))

  override def doExecute(): RDD[InternalRow] = {
    val numOutputRows = longMetric("numOutputRows")
    val numInputBatches = longMetric("numInputBatches")
    // This avoids calling `output` in the RDD closure, so that we don't need to include the entire
    // plan (this) in the closure.
    val localOutput = this.output
    child.executeColumnar().mapPartitionsInternal { batches =>
      val toUnsafe = UnsafeProjection.create(localOutput, localOutput)
      batches.flatMap { batch =>
        numInputBatches += 1
        numOutputRows += batch.numRows()

        // This is the original Spark code that creates an iterator over `ColumnarBatch`
        // to provide `Iterator[InternalRow]`. The implementation uses a `ColumnarBatchRow`
        // instance that contains an array of `ColumnVector` which will be instances of
        // `CometVector`, which in turn is a wrapper around Arrow's `ValueVector`.
        // We can refactor this code to have more efficient interactions with `CometVector`
        // so that we can bulk load data from JNI rather than fetch one value at a time.
        batch.rowIterator().asScala.map(toUnsafe)
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): CometColumnarToRowExec =
    copy(child = newChild)
}
