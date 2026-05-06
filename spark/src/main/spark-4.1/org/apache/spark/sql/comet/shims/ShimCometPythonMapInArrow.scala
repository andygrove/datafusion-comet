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

package org.apache.spark.sql.comet.shims

import org.apache.spark.{JobArtifactSet, TaskContext}
import org.apache.spark.api.python.ChainedPythonFunctions
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, PythonUDF}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.execution.python.{ArrowPythonRunner, MapInArrowExec, MapInPandasExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

trait ShimCometPythonMapInArrow {

  protected def matchMapInArrow(
      plan: SparkPlan): Option[(Expression, Seq[Attribute], SparkPlan, Boolean, Int)] =
    plan match {
      case p: MapInArrowExec =>
        Some((p.func, p.output, p.child, p.isBarrier, p.func.asInstanceOf[PythonUDF].evalType))
      case _ => None
    }

  protected def matchMapInPandas(
      plan: SparkPlan): Option[(Expression, Seq[Attribute], SparkPlan, Boolean, Int)] =
    plan match {
      case p: MapInPandasExec =>
        Some((p.func, p.output, p.child, p.isBarrier, p.func.asInstanceOf[PythonUDF].evalType))
      case _ => None
    }

  protected def currentJobArtifactUUID(): Option[String] =
    JobArtifactSet.getCurrentJobArtifactState.map(_.uuid)

  protected def largeVarTypes(conf: SQLConf): Boolean = conf.arrowUseLargeVarTypes

  protected def getPythonRunnerConfMap(conf: SQLConf): Map[String, String] =
    ArrowPythonRunner.getPythonRunnerConfMap(conf)

  protected def computeArrowPython(
      pythonUDF: PythonUDF,
      evalType: Int,
      argOffsets: Array[Array[Int]],
      schema: StructType,
      timeZoneId: String,
      largeVarTypes: Boolean,
      pythonRunnerConf: Map[String, String],
      pythonMetrics: Map[String, SQLMetric],
      jobArtifactUUID: Option[String],
      batchIter: Iterator[Iterator[InternalRow]],
      partitionId: Int,
      context: TaskContext): Iterator[ColumnarBatch] = {
    val chainedFunc =
      Seq((ChainedPythonFunctions(Seq(pythonUDF.func)), pythonUDF.resultId.id))
    new ArrowPythonRunner(
      chainedFunc,
      evalType,
      argOffsets,
      schema,
      timeZoneId,
      largeVarTypes,
      pythonRunnerConf,
      pythonMetrics,
      jobArtifactUUID,
      None,
      None).compute(batchIter, partitionId, context)
  }
}
