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

package org.apache.comet.parquet

import java.sql.Timestamp

import org.apache.spark.SparkException
import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.internal.SQLConf

import org.apache.comet.CometConf

/**
 * Verifies that the native reader correctly rejects reading INT96 timestamps as TimestampNTZ
 * (issue https://github.com/apache/datafusion-comet/issues/3720, SPARK-36182).
 *
 * INT96 Parquet timestamps encode UTC-adjusted instants. Reading them as TimestampNTZ would
 * silently reinterpret UTC instants as wall-clock values, producing incorrect results. Spark
 * itself raises (SPARK-36182) to prevent this, and Comet's native reader now does the same
 * via a guard in the schema adapter.
 */
class ParquetInt96NtzCorrectnessSuite extends CometTestBase {
  import testImplicits._

  test("native reader rejects reading INT96 as TimestampNTZ (SPARK-36182)") {
    val sessionTz = "America/Los_Angeles"
    val written = "2020-01-01 12:00:00"

    withSQLConf(
      SQLConf.SESSION_LOCAL_TIMEZONE.key -> sessionTz,
      SQLConf.PARQUET_OUTPUT_TIMESTAMP_TYPE.key -> "INT96",
      SQLConf.USE_V1_SOURCE_LIST.key -> "parquet",
      CometConf.COMET_NATIVE_SCAN_IMPL.key -> CometConf.SCAN_NATIVE_DATAFUSION) {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        Seq(Timestamp.valueOf(written)).toDF("ts").write.parquet(path)

        // Comet's native reader raises an error matching Spark's SPARK-36182 behavior
        intercept[SparkException] {
          spark.read.schema("ts timestamp_ntz").parquet(path).collect()
        }

        // Verify Spark also rejects this read
        withSQLConf(CometConf.COMET_ENABLED.key -> "false") {
          intercept[SparkException] {
            spark.read.schema("ts timestamp_ntz").parquet(path).collect()
          }
        }
      }
    }
  }

  test("INT96 read as TimestampLTZ works correctly") {
    val sessionTz = "America/Los_Angeles"
    val written = "2020-01-01 12:00:00"

    withSQLConf(
      SQLConf.SESSION_LOCAL_TIMEZONE.key -> sessionTz,
      SQLConf.PARQUET_OUTPUT_TIMESTAMP_TYPE.key -> "INT96",
      SQLConf.USE_V1_SOURCE_LIST.key -> "parquet",
      CometConf.COMET_NATIVE_SCAN_IMPL.key -> CometConf.SCAN_NATIVE_DATAFUSION) {
      withTempPath { dir =>
        val path = dir.getCanonicalPath
        Seq(Timestamp.valueOf(written)).toDF("ts").write.parquet(path)

        // Reading INT96 as the default TimestampType (LTZ) should produce correct results
        checkSparkAnswerAndOperator(spark.read.parquet(path))
      }
    }
  }
}
