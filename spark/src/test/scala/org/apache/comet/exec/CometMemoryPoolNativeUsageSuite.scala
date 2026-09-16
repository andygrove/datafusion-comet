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

import org.apache.spark.SparkConf
import org.apache.spark.sql.CometTestBase

import org.apache.comet.CometConf

/**
 * Exercises the off-heap memory pools' observation of real native memory usage.
 *
 * The budget is `spark.memory.offHeap.size`, which Spark fixes when the session starts, so these
 * tests need a session of their own rather than a `withSQLConf` override. The size below is
 * smaller than the memory Comet's native code holds once a query is running, so every reservation
 * crosses the budget and the observer has something to report. Deliberately not achieved by
 * lowering `spark.comet.exec.memoryPool.fraction`: that bounds what Comet may reserve, not what
 * the observer measures.
 */
class CometMemoryPoolNativeUsageSuite extends CometTestBase {

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.memory.offHeap.size", "2m")
    conf
  }

  /** A sort, so that an operator actually reserves. */
  private def sortSmallInput(): Unit =
    spark.range(0, 1000).selectExpr("id", "id % 7 AS m").sort("m", "id").collect()

  /**
   * The observer must never withhold a reservation, only report on it.
   *
   * The budget here is far too small for the sort, so the query fails either way. What this pins
   * down is *who* refuses it: the request has to reach Spark's ledger and fail there. An earlier
   * revision of this pool could refuse the reservation itself, which was removed because
   * measurement showed it did not keep real usage under the budget while it did penalise the
   * operators that reserve. If refusal is ever reintroduced, this test fails.
   */
  test("the pool observes real native usage without withholding reservations") {
    Seq("fair_unified", "greedy_unified").foreach { poolType =>
      withSQLConf(CometConf.COMET_OFFHEAP_MEMORY_POOL_TYPE.key -> poolType) {
        val messages = causeChain(intercept[Throwable](sortSmallInput()))
          .map(t => s"${t.getClass.getName}: ${t.getMessage}")
        assert(
          messages.exists(_.contains("failed to acquire")),
          s"expected $poolType to pass the reservation to Spark's ledger, but got:\n  " +
            messages.mkString("\n  "))
      }
    }
  }
}
