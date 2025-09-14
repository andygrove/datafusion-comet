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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.{OffHeapColumnVector, OnHeapColumnVector, WritableColumnVector}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.apache.comet.CometConf
import org.apache.comet.vector.ArrowToSparkConverter

/**
 * Physical operator that converts Arrow columnar batches (from Comet operators) to Spark columnar
 * batches (OnHeapColumnVector or OffHeapColumnVector). This allows InMemoryTableScanExec to cache
 * columnar data directly without performing columnar-to-row conversions.
 *
 * @param child
 *   The child plan that produces Arrow columnar batches
 * @param useOffHeap
 *   Whether to use off-heap memory for Spark column vectors
 */
case class CometColumnarToSparkColumnarExec(
    child: SparkPlan,
    useOffHeap: Boolean = SQLConf.get.offHeapColumnVectorEnabled)
    extends SparkPlan
    with CometPlan {

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override def supportsColumnar: Boolean = true

  override def children: Seq[SparkPlan] = Seq(child)

  override def nodeName: String = "CometColumnarToSparkColumnar"

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of input rows"),
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "conversionTime" -> SQLMetrics.createNanoTimingMetric(
      sparkContext,
      "time converting Arrow to Spark columnar"))

  override protected def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException(
      s"$nodeName does not support row-based execution. " +
        "It should only be used when the parent operator produces columnar output.")
  }

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numInputBatches = longMetric("numInputBatches")
    val numOutputBatches = longMetric("numOutputBatches")
    val numInputRows = longMetric("numInputRows")
    val numOutputRows = longMetric("numOutputRows")
    val conversionTime = longMetric("conversionTime")

    val enabledConf = CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.get(conf)
    if (!enabledConf) {
      // If conversion is disabled, pass through the Arrow batches unchanged
      child.executeColumnar()
    } else {
      child.executeColumnar().mapPartitionsInternal { arrowBatches =>
        new Iterator[ColumnarBatch] {
          override def hasNext: Boolean = arrowBatches.hasNext

          override def next(): ColumnarBatch = {
            val startTime = System.nanoTime()
            val arrowBatch = arrowBatches.next()
            numInputBatches += 1
            numInputRows += arrowBatch.numRows()

            try {
              val sparkBatch = convertArrowToSparkColumnar(arrowBatch)
              numOutputBatches += 1
              numOutputRows += sparkBatch.numRows()
              conversionTime += (System.nanoTime() - startTime)
              sparkBatch
            } finally {
              // Close the original Arrow batch to free memory
              arrowBatch.close()
            }
          }
        }
      }
    }
  }

  /**
   * Converts an Arrow columnar batch to a Spark columnar batch.
   *
   * @param arrowBatch
   *   The input Arrow columnar batch
   * @return
   *   A new Spark columnar batch with the same data
   */
  private def convertArrowToSparkColumnar(arrowBatch: ColumnarBatch): ColumnarBatch = {
    val numRows = arrowBatch.numRows()
    val numCols = arrowBatch.numCols()
    val schema = child.schema

    // Create Spark column vectors
    val sparkColumns = new Array[WritableColumnVector](numCols)

    try {
      for (i <- 0 until numCols) {
        val dataType = schema(i).dataType
        sparkColumns(i) = if (useOffHeap) {
          new OffHeapColumnVector(numRows, dataType)
        } else {
          new OnHeapColumnVector(numRows, dataType)
        }

        // Copy data from Arrow to Spark column vector
        val arrowColumn = arrowBatch.column(i)
        ArrowToSparkConverter.copyArrowToSparkVector(
          arrowColumn,
          sparkColumns(i),
          dataType,
          numRows)
      }

      // Create the new Spark columnar batch
      new ColumnarBatch(
        sparkColumns
          .asInstanceOf[Array[org.apache.spark.sql.vectorized.ColumnVector]],
        numRows)
    } catch {
      case e: Exception =>
        // Clean up allocated vectors on error
        sparkColumns.foreach { col =>
          if (col != null) col.close()
        }
        throw e
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(child = newChildren.head)
}

object CometColumnarToSparkColumnarExec {

  /**
   * Creates a CometColumnarToSparkColumnarExec operator if needed. This is used by query planning
   * rules to insert the conversion operator.
   *
   * @param child
   *   The child plan
   * @param conf
   *   SQL configuration
   * @return
   *   The child plan wrapped in conversion operator if needed, otherwise the child plan
   */
  def apply(child: SparkPlan, conf: SQLConf): SparkPlan = {
    if (shouldApplyConversion(child, conf)) {
      new CometColumnarToSparkColumnarExec(child, useOffHeap = conf.offHeapColumnVectorEnabled)
    } else {
      child
    }
  }

  /**
   * Determines if conversion should be applied based on the child plan and configuration.
   *
   * @param plan
   *   The plan to check
   * @param conf
   *   SQL configuration
   * @return
   *   true if conversion should be applied
   */
  private def shouldApplyConversion(plan: SparkPlan, conf: SQLConf): Boolean = {
    CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.get(conf) && isArrowColumnarOutput(plan)
  }

  /**
   * Checks if a plan produces Arrow columnar output.
   *
   * @param plan
   *   The plan to check
   * @return
   *   true if the plan produces Arrow columnar batches
   */
  private def isArrowColumnarOutput(plan: SparkPlan): Boolean = {
    plan match {
      case _: CometNativeExec => true
      case _: CometScanExec => true
      case _: CometSparkToColumnarExec => true
      case _ => false
    }
  }
}
