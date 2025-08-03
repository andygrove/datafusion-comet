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

import java.io.File

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.{ContextAwareIterator, JobArtifactSet, SparkEnv, TaskContext}
import org.apache.spark.api.python.ChainedPythonFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, Expression, MutableProjection, PythonUDF}
import org.apache.spark.sql.catalyst.trees.UnaryLike
import org.apache.spark.sql.comet.CometExec
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.Utils

/**
 * A physical plan that evaluates a [[PythonUDF]].
 */
case class CometArrowEvalPythonExec(
    override val originalPlan: SparkPlan,
    udfs: Seq[PythonUDF],
    resultAttrs: Seq[Attribute],
    child: SparkPlan,
    evalType: Int)
    extends CometExec
    with UnaryLike[SparkPlan]
    with PythonSQLMetrics {

  conf.arrowMaxRecordsPerBatch
  conf.sessionLocalTimeZone
  conf.arrowUseLargeVarTypes
  ArrowPythonRunner.getPythonRunnerConfMap(conf)
  private[this] val jobArtifactUUID = JobArtifactSet.getCurrentJobArtifactState.map(_.uuid)

  override def supportsColumnar: Boolean = true

  override def output: Seq[Attribute] = child.output ++ resultAttrs

  override def producedAttributes: AttributeSet = AttributeSet(resultAttrs)

  private def collectFunctions(udf: PythonUDF): (ChainedPythonFunctions, Seq[Expression]) = {
    udf.children match {
      case Seq(u: PythonUDF) =>
        val (chained, children) = collectFunctions(u)
        (ChainedPythonFunctions(chained.funcs ++ Seq(udf.func)), children)
      case children =>
        // There should not be any other UDFs, or the children can't be evaluated directly.
        assert(children.forall(!_.exists(_.isInstanceOf[PythonUDF])))
        (ChainedPythonFunctions(Seq(udf.func)), udf.children)
    }
  }

  override protected def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val inputRDD = child.executeColumnar()

    inputRDD.mapPartitions { iter =>
      val context = TaskContext.get()
      val contextAwareIterator = new ContextAwareIterator(context, iter)

      // The queue used to buffer input rows so we can drain it to
      // combine input with output from Python.
      val queue = HybridRowQueue(
        context.taskMemoryManager(),
        new File(Utils.getLocalDir(SparkEnv.get.conf)),
        child.output.length)
      context.addTaskCompletionListener[Unit] { ctx =>
        queue.close()
      }

      val (pyFuncs, inputs) = udfs.map(collectFunctions).unzip

      // flatten all the arguments
      val allInputs = new ArrayBuffer[Expression]
      val dataTypes = new ArrayBuffer[DataType]
      val argOffsets = inputs.map { input =>
        input.map { e =>
          if (allInputs.exists(_.semanticEquals(e))) {
            allInputs.indexWhere(_.semanticEquals(e))
          } else {
            allInputs += e
            dataTypes += e.dataType
            allInputs.length - 1
          }
        }.toArray
      }.toArray
      val projection = MutableProjection.create(allInputs.toSeq, child.output)
      projection.initialize(context.partitionId())
      val schema = StructType(dataTypes.zipWithIndex.map { case (dt, i) =>
        StructField(s"_$i", dt)
      }.toArray)

      // TODO reinstate queue
      // Add rows to queue to join later with the result.
//      val projectedRowIter = contextAwareIterator.map { inputBatch =>
//        queue.add(inputBatch /*.asInstanceOf[UnsafeRow]*/)
//        //projection(inputBatch)
//        inputBatch
//      }

      val outputBatchIterator =
        evaluate(pyFuncs, argOffsets, contextAwareIterator, schema, context)

//      val joined = new JoinedRow
//      val resultProj = UnsafeProjection.create(output, output)

      outputBatchIterator

    // TODO reinstate queue
    /*
      .map { outputRow => resultProj(joined(queue.remove(), outputRow))
      }
     */

    }
  }

  def evaluate(
      funcs: Seq[ChainedPythonFunctions],
      argOffsets: Array[Array[Int]],
      iter: Iterator[ColumnarBatch],
      schema: StructType,
      context: TaskContext): Iterator[ColumnarBatch] = {

//    val outputTypes = output.drop(child.output.length).map(_.dataType)

    // DO NOT use iter.grouped(). See BatchIterator.
//    val batchIter = if (batchSize > 0) new BatchIterator(iter, batchSize) else Iterator(iter)

    val columnarBatchIter = new CometArrowPythonRunner(
      funcs,
      evalType,
      argOffsets,
//      schema,
//      sessionLocalTimeZone,
//      largeVarTypes,
//      pythonRunnerConf,
      pythonMetrics,
      jobArtifactUUID).compute(iter, context.partitionId(), context)

    columnarBatchIter
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    copy(child = newChild)
}
