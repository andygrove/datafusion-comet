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

import org.apache.spark.sql.execution.SparkPlan

import org.apache.comet.ConfigEntry
import org.apache.comet.serde.OperatorOuterClass.Operator

/**
 * Trait for providing serialization logic for operators.
 */
trait CometOperatorHandler[T <: SparkPlan] {

  /**
   * Get the optional Comet configuration entry that is used to enable or disable native support
   * for this operator.
   */
  def enabledConfig: Option[ConfigEntry[Boolean]]

  /**
   * Determine the support level of the operator based on its attributes.
   *
   * @param operator
   *   The Spark operator.
   * @return
   *   Support level (Compatible, Incompatible, or Unsupported).
   */
  def getSupportLevel(operator: T): SupportLevel = Compatible(None)

  /**
   * Convert a Spark operator into a protocol buffer representation that can be passed into native
   * code.
   *
   * @param op
   *   The Spark operator.
   * @param builder
   *   The protobuf builder for the operator.
   * @param childOp
   *   Child operators that have already been converted to Comet.
   * @return
   *   Protocol buffer representation, or None if the operator could not be converted. In this
   *   case it is expected that the input operator will have been tagged with reasons why it could
   *   not be converted.
   */
  def convert(
      op: T,
      builder: Operator.Builder,
      childOp: Operator*): Option[OperatorOuterClass.Operator]

  /**
   * Create a Comet execution plan node from a Spark operator.
   *
   * This method combines protobuf conversion with Comet operator creation, delegating the entire
   * transformation process to the serde implementation.
   *
   * @param op
   *   The Spark operator to convert.
   * @param nativeOp
   *   The protobuf representation of the operator.
   * @param child
   *   The converted Comet child operator(s).
   * @return
   *   A CometExec wrapping the native operator, or None if conversion is not applicable.
   */
  def createExec(op: T, nativeOp: Operator, child: SparkPlan*): Option[SparkPlan] = {
    // Default implementation returns None, indicating this serde does not support
    // the delegated execution plan creation. Operators that want to use the new
    // pattern should override this method.
    None
  }

}
