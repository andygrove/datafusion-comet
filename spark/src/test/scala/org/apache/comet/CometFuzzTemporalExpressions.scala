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

import org.apache.spark.sql.types.DataTypes

class CometFuzzTemporalExpressions extends CometFuzzTestBase {

  for (func <- Seq("date_add", "date_sub")) {
    test(s"datemake  arithmetic - $func") {
      val df = spark.read.parquet(filename)
      df.createOrReplaceTempView("t1")
      val dateCol = df.schema.fields.filter(_.dataType == DataTypes.DateType).map(_.name).head
      val intCol = df.schema.fields.filter(_.dataType == DataTypes.IntegerType).map(_.name).head
      val sql =
        s"""SELECT $dateCol, $func($dateCol, $intCol)
           |FROM t1
           |WHERE $intCol BETWEEN -100000 AND 100000
           |ORDER BY $dateCol, $intCol""".stripMargin
      if (usingDataSourceExec) {
        checkSparkAnswerAndOperator(sql)
      } else {
        checkSparkAnswer(sql)
      }
    }
  }
}
