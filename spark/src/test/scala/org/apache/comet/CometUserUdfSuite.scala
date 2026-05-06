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

package org.apache.comet

import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.types.LongType

import org.apache.comet.udf.CometUdfRegistry

class CometUserUdfSuite extends CometTestBase with AdaptiveSparkPlanHelper {

  override def afterEach(): Unit = {
    CometUdfRegistry.clear()
    super.afterEach()
  }

  private def registerDoubleInt(): Unit = {
    CometUdfRegistry.register(
      spark,
      "double_int",
      "org.apache.comet.udf.testing.DoubleIntUdf",
      (x: Int) => x.toLong * 2L,
      LongType,
      nullable = true)
  }

  test("user CometUDF - basic integer doubling") {
    registerDoubleInt()
    withTable("t") {
      sql("CREATE TABLE t (x INT) USING parquet")
      sql("INSERT INTO t VALUES (1), (2), (3), (NULL), (100)")
      checkSparkAnswerAndOperator(sql("SELECT double_int(x) FROM t"))
    }
  }

  test("user CometUDF - unregistered UDF falls back to Spark") {
    spark.udf.register("triple_int", (x: Int) => x * 3)

    withTable("t") {
      sql("CREATE TABLE t (x INT) USING parquet")
      sql("INSERT INTO t VALUES (1), (2), (3)")
      checkSparkAnswerAndFallbackReason(
        sql("SELECT triple_int(x) FROM t"),
        "ScalaUDF 'triple_int' is not registered in CometUdfRegistry")
    }
  }

  test("user CometUDF - multiple arguments") {
    registerDoubleInt()
    withTable("t") {
      sql("CREATE TABLE t (x INT, y INT) USING parquet")
      sql("INSERT INTO t VALUES (10, 20), (NULL, 5), (3, NULL)")
      checkSparkAnswerAndOperator(sql("SELECT double_int(x), double_int(y) FROM t"))
    }
  }

  test("user CometUDF - with filter") {
    registerDoubleInt()
    withTable("t") {
      sql("CREATE TABLE t (x INT) USING parquet")
      sql("INSERT INTO t VALUES (1), (2), (3), (4), (5)")
      checkSparkAnswerAndOperator(sql("SELECT double_int(x) FROM t WHERE x > 2"))
    }
  }

  test("CometUdfRegistry - register and lookup") {
    assert(!CometUdfRegistry.isRegistered("test_func"))
    CometUdfRegistry.register("test_func", "com.example.TestUdf", LongType, nullable = false)
    assert(CometUdfRegistry.isRegistered("test_func"))
    val entry = CometUdfRegistry.get("test_func")
    assert(entry.isDefined)
    assert(entry.get.className == "com.example.TestUdf")
    assert(entry.get.returnType == LongType)
    assert(!entry.get.nullable)
    CometUdfRegistry.remove("test_func")
    assert(!CometUdfRegistry.isRegistered("test_func"))
  }
}
