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

package org.apache.comet.rules

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.comet._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec

import org.apache.comet.CometConf

/**
 * Rule to insert CometColumnarToSparkColumnarExec operators when needed for caching. This rule
 * ensures that when InMemoryTableScanExec is used with Comet operators, the data is converted
 * from Arrow columnar format to Spark columnar format to avoid ColumnarToRow conversions.
 */
case class InsertColumnarToSparkColumnar(session: SparkSession) extends Rule[SparkPlan] {

  override val conf = session.sessionState.conf

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!CometConf.COMET_ENABLED.get(conf) ||
      !CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.get(conf)) {
      return plan
    }

    plan.transformUp {
      // When we have a ColumnarToRowExec above InMemoryTableScanExec,
      // and the input to the cache is from Comet operators,
      // we should insert our conversion operator
      case c2r @ ColumnarToRowExec(scan: InMemoryTableScanExec) if hasCometColumnarInput(scan) =>
        // Replace ColumnarToRowExec with identity since we'll keep data columnar
        scan

      // When InMemoryTableScanExec's cached plan outputs Arrow columnar batches,
      // we need to convert them to Spark columnar format for caching
      case scan: InMemoryTableScanExec if requiresConversion(scan) =>
        // Wrap the scan with our conversion operator
        CometColumnarToSparkColumnarExec(scan, conf.offHeapColumnVectorEnabled)

      // Handle InMemoryRelation that needs conversion during cache building
      case plan: SparkPlan if needsCacheConversion(plan) =>
        insertConversionForCaching(plan)
    }
  }

  /**
   * Checks if an InMemoryTableScanExec has Comet columnar input in its cached relation.
   */
  private def hasCometColumnarInput(scan: InMemoryTableScanExec): Boolean = {
    // Check if the cached relation was built from Comet operators
    scan.relation.cacheBuilder.cachedPlan match {
      case _: CometNativeExec => true
      case _: CometScanExec => true
      case _: CometSparkToColumnarExec => true
      case plan: SparkPlan if plan.supportsColumnar && containsCometOperator(plan) => true
      case _ => false
    }
  }

  /**
   * Checks if an InMemoryTableScanExec requires conversion from Arrow to Spark columnar.
   */
  private def requiresConversion(scan: InMemoryTableScanExec): Boolean = {
    // If the scan is outputting columnar data and it comes from Comet operators
    scan.supportsColumnar && hasCometColumnarInput(scan)
  }

  /**
   * Recursively checks if a plan contains any Comet operators.
   */
  private def containsCometOperator(plan: SparkPlan): Boolean = {
    plan match {
      case _: CometPlan => true
      case _ => plan.children.exists(containsCometOperator)
    }
  }

  /**
   * Checks if a plan needs conversion when building a cache.
   */
  private def needsCacheConversion(plan: SparkPlan): Boolean = {
    // This would need to be handled at the logical plan level during cache building
    // For now, we focus on the InMemoryTableScanExec conversion
    false
  }

  /**
   * Inserts conversion operator for cache building.
   */
  private def insertConversionForCaching(plan: SparkPlan): SparkPlan = {
    // This would need to be handled at the logical plan level during cache building
    // For now, we focus on the InMemoryTableScanExec conversion
    plan
  }
}
