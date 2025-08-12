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

import org.apache.spark.sql.execution.{GlobalLimitExec, LocalLimitExec}

import org.apache.comet.{CometConf, ConfigEntry}
import org.apache.comet.CometSparkSessionExtensions.withInfo
import org.apache.comet.serde.OperatorOuterClass.Operator

object CometLocalLimit extends CometOperatorSerde[LocalLimitExec] {

  override def enabledConfig: Option[ConfigEntry[Boolean]] =
    Some(CometConf.COMET_EXEC_LOCAL_LIMIT_ENABLED)

  override def convert(
      limit: LocalLimitExec,
      builder: Operator.Builder,
      childOp: OperatorOuterClass.Operator*): Option[OperatorOuterClass.Operator] = {
    if (childOp.nonEmpty) {
      // LocalLimit doesn't use offset, but it shares same operator serde class.
      // Just set it to zero.
      val limitBuilder = OperatorOuterClass.Limit
        .newBuilder()
        .setLimit(limit.limit)
        .setOffset(0)
      Some(builder.setLimit(limitBuilder).build())
    } else {
      withInfo(limit, "No child operator")
      None
    }

  }
}

object CometGlobalLimit extends CometOperatorSerde[GlobalLimitExec] {

  override def enabledConfig: Option[ConfigEntry[Boolean]] =
    Some(CometConf.COMET_EXEC_GLOBAL_LIMIT_ENABLED)

  override def convert(
      op: GlobalLimitExec,
      builder: Operator.Builder,
      childOp: OperatorOuterClass.Operator*): Option[OperatorOuterClass.Operator] = {
    // TODO: We don't support negative limit for now.
    if (childOp.nonEmpty && op.limit >= 0) {
      val limitBuilder = OperatorOuterClass.Limit.newBuilder()

      // TODO: Spark 3.3 might have negative limit (-1) for Offset usage.
      // When we upgrade to Spark 3.3., we need to address it here.
      limitBuilder.setLimit(op.limit)

      Some(builder.setLimit(limitBuilder).build())
    } else {
      withInfo(op, "No child operator")
      None
    }
  }
}
