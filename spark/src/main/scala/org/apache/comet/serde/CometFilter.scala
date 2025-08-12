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

package org.apache.comet.serde

import org.apache.spark.sql.catalyst.expressions.{Contains, EndsWith, Expression, StartsWith}
import org.apache.spark.sql.comet.{CometNativeScanExec, CometScanExec, CometScanWrapper}
import org.apache.spark.sql.execution.{FilterExec, SparkPlan}

import org.apache.comet.{CometConf, ConfigEntry}
import org.apache.comet.CometSparkSessionExtensions.withInfo
import org.apache.comet.serde.OperatorOuterClass.Operator
import org.apache.comet.serde.QueryPlanSerde.exprToProto

object CometFilter extends CometOperatorSerde[FilterExec] {

  override def enabledConfig: Option[ConfigEntry[Boolean]] =
    Some(CometConf.COMET_EXEC_FILTER_ENABLED)

  override def convert(
      op: FilterExec,
      builder: Operator.Builder,
      childOp: OperatorOuterClass.Operator*): Option[OperatorOuterClass.Operator] = {
    val cond = exprToProto(op.condition, op.child.output)

    if (cond.isDefined && childOp.nonEmpty) {
      // We need to determine whether to use DataFusion's FilterExec or Comet's
      // FilterExec. The difference is that DataFusion's implementation will sometimes pass
      // batches through whereas the Comet implementation guarantees that a copy is always
      // made, which is critical when using `native_comet` scans due to buffer re-use

      // TODO this could be optimized more to stop walking the tree on hitting
      //  certain operators such as join or aggregate which will copy batches
      def containsNativeCometScan(plan: SparkPlan): Boolean = {
        plan match {
          case w: CometScanWrapper => containsNativeCometScan(w.originalPlan)
          case scan: CometScanExec => scan.scanImpl == CometConf.SCAN_NATIVE_COMET
          case _: CometNativeScanExec => false
          case _ => plan.children.exists(containsNativeCometScan)
        }
      }

      // Some native expressions do not support operating on dictionary-encoded arrays, so
      // wrap the child in a CopyExec to unpack dictionaries first.
      def wrapChildInCopyExec(condition: Expression): Boolean = {
        condition.exists(expr => {
          expr.isInstanceOf[StartsWith] || expr.isInstanceOf[EndsWith] || expr
            .isInstanceOf[Contains]
        })
      }

      val filterBuilder = OperatorOuterClass.Filter
        .newBuilder()
        .setPredicate(cond.get)
        .setUseDatafusionFilter(!containsNativeCometScan(op))
        .setWrapChildInCopyExec(wrapChildInCopyExec(op.condition))
      Some(builder.setFilter(filterBuilder).build())
    } else {
      withInfo(op, op.condition, op.child)
      None
    }

  }
}
