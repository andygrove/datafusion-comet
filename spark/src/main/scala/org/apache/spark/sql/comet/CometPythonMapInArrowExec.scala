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
import org.apache.spark.api.python.ChainedPythonFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.PythonUDF
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.python.{ArrowPythonRunner, BatchIterator, PythonSQLMetrics}
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
    with PythonSQLMetrics {

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

    val pythonRunnerConf = ArrowPythonRunner.getPythonRunnerConfMap(conf)
    val pythonFunction = func.asInstanceOf[PythonUDF].func
    val chainedFunc = Seq(ChainedPythonFunctions(Seq(pythonFunction)))
    val localOutput = output
    val localChildSchema = child.schema
    val batchSize = conf.arrowMaxRecordsPerBatch
    val sessionLocalTimeZone = conf.sessionLocalTimeZone
    val largeVarTypes = conf.arrowUseLargeVarTypes
    val localPythonEvalType = pythonEvalType
    val localPythonMetrics = pythonMetrics
    val jobArtifactUUID =
      org.apache.spark.JobArtifactSet.getCurrentJobArtifactState.map(_.uuid)

    val inputRDD = child.executeColumnar()

    inputRDD.mapPartitionsInternal { batches =>
      val context = TaskContext.get()
      val argOffsets = Array(Array(0))

      // Convert columnar batches to rows using lightweight rowIterator
      // (avoids UnsafeProjection copy that ColumnarToRow would do)
      val rowIter = batches.flatMap { batch =>
        numInputRows += batch.numRows()
        batch.rowIterator().asScala
      }

      val contextAwareIterator = new ContextAwareIterator(context, rowIter)

      // Wrap rows as a struct, matching MapInBatchEvaluatorFactory behavior
      val wrappedIter = contextAwareIterator.map(InternalRow(_))

      val batchIter =
        if (batchSize > 0) new BatchIterator(wrappedIter, batchSize) else Iterator(wrappedIter)

      val columnarBatchIter = new ArrowPythonRunner(
        chainedFunc,
        localPythonEvalType,
        argOffsets,
        org.apache.spark.sql.types
          .StructType(Array(org.apache.spark.sql.types.StructField("struct", localChildSchema))),
        sessionLocalTimeZone,
        largeVarTypes,
        pythonRunnerConf,
        localPythonMetrics,
        jobArtifactUUID).compute(batchIter, context.partitionId(), context)

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
  }

  override protected def withNewChildInternal(newChild: SparkPlan): CometPythonMapInArrowExec =
    copy(child = newChild)
}
