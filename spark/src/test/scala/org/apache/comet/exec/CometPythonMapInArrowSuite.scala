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
import org.apache.spark.sql.comet.CometPythonMapInArrowExec

import org.apache.comet.CometConf

class CometPythonMapInArrowSuite extends CometTestBase {

  test("plan with CometScan has columnar support for Python UDF optimization") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_PYTHON_MAP_IN_ARROW_ENABLED.key -> "true") {
      withParquetTable(
        (1 to 10).map(i => (i.toDouble, s"str_$i")),
        "testTable",
        withDictionary = false) {
        val df = spark.sql("SELECT * FROM testTable")
        val plan = df.queryExecution.executedPlan
        val cometScans = plan.collect { case s if s.supportsColumnar => s }
        assert(cometScans.nonEmpty, "Expected columnar operators that can feed Python UDFs")
      }
    }
  }

  test("config disables Python map in arrow optimization") {
    withSQLConf(
      CometConf.COMET_ENABLED.key -> "true",
      CometConf.COMET_EXEC_ENABLED.key -> "true",
      CometConf.COMET_PYTHON_MAP_IN_ARROW_ENABLED.key -> "false") {
      withParquetTable(
        (1 to 10).map(i => (i.toDouble, s"str_$i")),
        "testTable",
        withDictionary = false) {
        val df = spark.sql("SELECT * FROM testTable")
        val plan = df.queryExecution.executedPlan
        // With the feature disabled, no CometPythonMapInArrowExec should appear
        val cometPythonExecs =
          plan.collect { case e: CometPythonMapInArrowExec => e }
        assert(
          cometPythonExecs.isEmpty,
          "CometPythonMapInArrowExec should not appear when disabled")
      }
    }
  }
}
