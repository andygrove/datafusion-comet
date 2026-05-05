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

package org.apache.comet.udf

import org.apache.arrow.vector.{BigIntVector, IntVector, ValueVector}

import org.apache.comet.CometArrowAllocator

/**
 * Test CometUDF that doubles an integer input. Used by CometUserUdfSuite.
 */
class DoubleIntUdf extends CometUDF {

  override def evaluate(inputs: Array[ValueVector]): ValueVector = {
    val input = inputs(0).asInstanceOf[IntVector]
    val rowCount = input.getValueCount

    val result = new BigIntVector("result", CometArrowAllocator)
    result.allocateNew(rowCount)

    var i = 0
    while (i < rowCount) {
      if (input.isNull(i)) {
        result.setNull(i)
      } else {
        result.set(i, input.get(i).toLong * 2L)
      }
      i += 1
    }
    result.setValueCount(rowCount)
    result
  }
}
