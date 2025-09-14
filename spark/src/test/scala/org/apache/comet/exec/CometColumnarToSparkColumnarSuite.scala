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

package org.apache.comet.exec

import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.comet.CometColumnarToSparkColumnarExec
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan}
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.internal.SQLConf

import org.apache.comet.CometConf

/**
 * Test suite for CometColumnarToSparkColumnar functionality.
 */
class CometColumnarToSparkColumnarSuite extends CometTestBase {

  test("Arrow to Spark columnar conversion for cached table") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("test_table") {
        // Create test data
        val df = spark
          .range(1000)
          .selectExpr("id", "id % 10 as value", "cast(id as string) as str")
        df.createOrReplaceTempView("test_table")

        // Cache the table
        spark.catalog.cacheTable("test_table")

        // Query the cached table
        val cachedDf = spark.sql("SELECT * FROM test_table WHERE value > 5")

        // Check that the plan contains our conversion operator
        val plan = cachedDf.queryExecution.executedPlan
        assert(
          containsOperator(plan, classOf[CometColumnarToSparkColumnarExec]),
          s"Plan should contain CometColumnarToSparkColumnarExec:\n${plan.treeString}")

        // Verify results
        checkSparkAnswerAndOperator(cachedDf)

        // Uncache
        spark.catalog.uncacheTable("test_table")
      }
    }
  }

  test("Conversion disabled when config is false") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "false",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("test_table") {
        val df = spark.range(100).selectExpr("id", "id * 2 as value")
        df.createOrReplaceTempView("test_table")

        spark.catalog.cacheTable("test_table")

        val cachedDf = spark.sql("SELECT * FROM test_table")
        val plan = cachedDf.queryExecution.executedPlan

        // Should not contain our conversion operator when disabled
        assert(
          !containsOperator(plan, classOf[CometColumnarToSparkColumnarExec]),
          s"Plan should not contain CometColumnarToSparkColumnarExec when disabled:\n${plan.treeString}")

        spark.catalog.uncacheTable("test_table")
      }
    }
  }

  test("Complex data types conversion") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("complex_table") {
        // Create test data with various data types
        val df = spark
          .range(100)
          .selectExpr(
            "id",
            "cast(id as tinyint) as byte_col",
            "cast(id as smallint) as short_col",
            "cast(id as int) as int_col",
            "cast(id as bigint) as long_col",
            "cast(id as float) as float_col",
            "cast(id as double) as double_col",
            "cast(id as string) as string_col",
            "cast(id as boolean) as bool_col",
            "cast(id as decimal(10,2)) as decimal_col",
            "date_add('2024-01-01', cast(id as int)) as date_col",
            "cast('2024-01-01 00:00:00' as timestamp) + interval 1 hour * id as timestamp_col")

        df.createOrReplaceTempView("complex_table")
        spark.catalog.cacheTable("complex_table")

        val cachedDf = spark.sql("SELECT * FROM complex_table WHERE id < 50")

        // Verify the plan contains our operator
        val plan = cachedDf.queryExecution.executedPlan
        assert(containsOperator(plan, classOf[CometColumnarToSparkColumnarExec]))

        // Verify results match
        checkSparkAnswerAndOperator(cachedDf)

        spark.catalog.uncacheTable("complex_table")
      }
    }
  }

  test("Aggregation over cached table with columnar conversion") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("agg_table") {
        val df = spark.range(1000).selectExpr("id", "id % 100 as group_id", "id * 2.5 as value")

        df.createOrReplaceTempView("agg_table")
        spark.catalog.cacheTable("agg_table")

        val aggDf = spark.sql("""
          SELECT group_id,
                 COUNT(*) as cnt,
                 SUM(value) as sum_val,
                 AVG(value) as avg_val,
                 MAX(value) as max_val,
                 MIN(value) as min_val
          FROM agg_table
          GROUP BY group_id
          HAVING COUNT(*) > 5
          ORDER BY group_id
        """)

        // Check plan and results
        val plan = aggDf.queryExecution.executedPlan
        assert(containsOperator(plan, classOf[InMemoryTableScanExec]))
        checkSparkAnswerAndOperator(aggDf)

        spark.catalog.uncacheTable("agg_table")
      }
    }
  }

  test("Join with cached tables using columnar conversion") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("left_table", "right_table") {
        val leftDf = spark.range(100).selectExpr("id", "id * 2 as value")
        val rightDf = spark.range(50, 150).selectExpr("id", "id * 3 as value")

        leftDf.createOrReplaceTempView("left_table")
        rightDf.createOrReplaceTempView("right_table")

        // Cache both tables
        spark.catalog.cacheTable("left_table")
        spark.catalog.cacheTable("right_table")

        val joinDf = spark.sql("""
          SELECT l.id, l.value as left_val, r.value as right_val
          FROM left_table l
          JOIN right_table r ON l.id = r.id
          WHERE l.id < 100
        """)

        // Verify plan and results
        checkSparkAnswerAndOperator(joinDf)

        spark.catalog.uncacheTable("left_table")
        spark.catalog.uncacheTable("right_table")
      }
    }
  }

  test("Verify no ColumnarToRow when using columnar conversion") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("no_c2r_table") {
        val df = spark.range(100).selectExpr("id", "id % 10 as value")
        df.createOrReplaceTempView("no_c2r_table")

        spark.catalog.cacheTable("no_c2r_table")

        // Query that would normally have ColumnarToRow
        val cachedDf = spark.sql("SELECT * FROM no_c2r_table")
        val plan = cachedDf.queryExecution.executedPlan

        // Check that there's no ColumnarToRow immediately after InMemoryTableScanExec
        def checkNoColumnarToRowAfterScan(plan: SparkPlan): Boolean = {
          plan match {
            case ColumnarToRowExec(scan: InMemoryTableScanExec) => false
            case _ => plan.children.forall(checkNoColumnarToRowAfterScan)
          }
        }

        assert(
          checkNoColumnarToRowAfterScan(plan),
          s"Plan should not have ColumnarToRow directly after InMemoryTableScanExec:\n${plan.treeString}")

        spark.catalog.uncacheTable("no_c2r_table")
      }
    }
  }

  test("Performance metrics collection") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_COLUMNAR_TO_SPARK_ENABLED.key -> "true",
      SQLConf.CACHE_VECTORIZED_READER_ENABLED.key -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "false") {

      withTempView("metrics_table") {
        val df = spark.range(10000).selectExpr("id", "id * 2 as value")
        df.createOrReplaceTempView("metrics_table")

        spark.catalog.cacheTable("metrics_table")

        val cachedDf = spark.sql("SELECT * FROM metrics_table WHERE value > 100")
        cachedDf.collect() // Execute the query

        val plan = cachedDf.queryExecution.executedPlan

        // Find the conversion operator and check its metrics
        val conversionOp = findOperator(plan, classOf[CometColumnarToSparkColumnarExec])
        conversionOp.foreach { op =>
          val metrics = op.metrics

          assert(metrics.contains("numInputBatches"))
          assert(metrics.contains("numOutputBatches"))
          assert(metrics.contains("numInputRows"))
          assert(metrics.contains("numOutputRows"))
          assert(metrics.contains("conversionTime"))

          // Verify metrics have been updated
          assert(metrics("numInputBatches").value > 0, "numInputBatches should be greater than 0")
          assert(
            metrics("numOutputBatches").value > 0,
            "numOutputBatches should be greater than 0")
          assert(metrics("conversionTime").value > 0, "conversionTime should be greater than 0")
        }

        spark.catalog.uncacheTable("metrics_table")
      }
    }
  }

  /**
   * Helper method to check if a plan contains a specific operator type.
   */
  private def containsOperator(plan: SparkPlan, opClass: Class[_]): Boolean = {
    plan.find(opClass.isInstance).isDefined
  }

  /**
   * Helper method to find a specific operator in the plan.
   */
  private def findOperator[T <: SparkPlan](plan: SparkPlan, opClass: Class[T]): Option[T] = {
    plan.find(opClass.isInstance).map(_.asInstanceOf[T])
  }
}
