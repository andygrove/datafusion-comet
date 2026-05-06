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

import org.apache.spark.{ContextAwareIterator, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.PythonUDF
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.comet.shims.ShimCometPythonMapInArrow
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.python.{BatchIterator, PythonSQLMetrics}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.vectorized.{ArrowColumnVector, ColumnarBatch}

/**
 * An optimized version of Spark's MapInBatchExec (PythonMapInArrowExec / MapInPandasExec) that
 * accepts columnar input directly from Comet operators, avoiding unnecessary Arrow -> Row ->
 * Arrow conversions.
 *
 * Normal Spark flow: CometNativeExec (Arrow) -> ColumnarToRow -> PythonMapInArrowExec
 * (internally: rows -> Arrow -> Python -> Arrow -> rows)
 *
 * Optimized flow: CometNativeExec (Arrow) -> CometPythonMapInArrowExec (batch.rowIterator() ->
 * Arrow -> Python -> Arrow columnar output)
 *
 * This eliminates:
 *   1. The UnsafeProjection in ColumnarToRow (expensive copy) 2. The output Arrow->Row conversion
 *      (keeps Python output as ColumnarBatch)
 */
case class CometPythonMapInArrowExec(
    func: Expression,
    output: Seq[Attribute],
    child: SparkPlan,
    isBarrier: Boolean,
    pythonEvalType: Int)
    extends UnaryExecNode
    with PythonSQLMetrics
    with ShimCometPythonMapInArrow {

  override def supportsColumnar: Boolean = true

  override def producedAttributes: AttributeSet = AttributeSet(output)

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "number of output batches"),
    "numInputRows" -> SQLMetrics.createMetric(sparkContext, "number of input rows")) ++
    pythonMetrics

  override def doExecute(): RDD[InternalRow] = {
    ColumnarToRowExec(this).doExecute()
  }

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val numInputRows = longMetric("numInputRows")

    val pythonUDF = func.asInstanceOf[PythonUDF]
    val localOutput = output
    val localChildSchema = child.schema
    val batchSize = conf.arrowMaxRecordsPerBatch
    val sessionLocalTimeZone = conf.sessionLocalTimeZone
    val useLargeVarTypes = largeVarTypes(conf)
    val pythonRunnerConf = getPythonRunnerConfMap(conf)
    val localPythonEvalType = pythonEvalType
    val localPythonMetrics = pythonMetrics
    val jobArtifactUUID = currentJobArtifactUUID()

    val inputRDD = child.executeColumnar()

    // Run on every partition. Identical to what MapInBatchExec does, except the input
    // is columnar; we intentionally avoid the UnsafeProjection copy that ColumnarToRow
    // would do.
    def processPartition(batches: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = {
      val context = TaskContext.get()
      val argOffsets = Array(Array(0))

      val rowIter = batches.flatMap { batch =>
        numInputRows += batch.numRows()
        batch.rowIterator().asScala
      }

      val contextAwareIterator = new ContextAwareIterator(context, rowIter)

      // Wrap rows as a struct, matching MapInBatchEvaluatorFactory behavior
      val wrappedIter = contextAwareIterator.map(InternalRow(_))

      val batchIter =
        if (batchSize > 0) new BatchIterator(wrappedIter, batchSize) else Iterator(wrappedIter)

      val columnarBatchIter = computeArrowPython(
        pythonUDF,
        localPythonEvalType,
        argOffsets,
        StructType(Array(StructField("struct", localChildSchema))),
        sessionLocalTimeZone,
        useLargeVarTypes,
        pythonRunnerConf,
        localPythonMetrics,
        jobArtifactUUID,
        batchIter,
        context.partitionId(),
        context)

      columnarBatchIter.map { batch =>
        // Python returns a StructType column; flatten to individual columns
        val structVector = batch.column(0).asInstanceOf[ArrowColumnVector]
        val outputVectors = localOutput.indices.map(structVector.getChild)
        val flattenedBatch = new ColumnarBatch(outputVectors.toArray)
        flattenedBatch.setNumRows(batch.numRows())
        numOutputRows += flattenedBatch.numRows()
        numOutputBatches += 1
        flattenedBatch
      }
    }

    // Preserve isBarrier semantics: when set, run inside a barrier stage so all tasks
    // are gang-scheduled and BarrierTaskContext.barrier() works inside the UDF.
    if (isBarrier) {
      inputRDD.barrier().mapPartitions(processPartition)
    } else {
      inputRDD.mapPartitionsInternal(processPartition)
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): CometPythonMapInArrowExec =
    copy(child = newChild)
}
