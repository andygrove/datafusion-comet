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

import java.util.concurrent.ConcurrentHashMap

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType

/**
 * Registry for user-defined CometUDF implementations. Users register their UDF class names here
 * so that the Comet serde layer can intercept matching Spark UDFs and route them to native
 * execution via the JVM UDF bridge.
 *
 * Usage:
 * {{{
 * // Register a CometUDF implementation for a Spark UDF
 * CometUdfRegistry.register(
 *   "my_func",                      // Spark UDF name (as used in spark.udf.register)
 *   "com.example.MyUdf",            // CometUDF implementation class
 *   BooleanType,                    // return type
 *   nullable = true                 // whether the result may contain nulls
 * )
 *
 * // Or use the convenience method that also registers the Spark UDF:
 * CometUdfRegistry.register(
 *   spark,
 *   "my_func",
 *   "com.example.MyUdf",
 *   sparkUdf,                       // the Spark UserDefinedFunction
 *   BooleanType,
 *   nullable = true
 * )
 * }}}
 */
object CometUdfRegistry {

  case class UdfEntry(className: String, returnType: DataType, nullable: Boolean)

  private val registry = new ConcurrentHashMap[String, UdfEntry]()

  /**
   * Register a CometUDF implementation for a named Spark UDF.
   *
   * @param name
   *   The UDF name as registered with Spark (via spark.udf.register)
   * @param className
   *   Fully-qualified class name implementing CometUDF
   * @param returnType
   *   The return DataType of the UDF
   * @param nullable
   *   Whether the result column may contain nulls
   */
  def register(name: String, className: String, returnType: DataType, nullable: Boolean): Unit = {
    registry.put(name, UdfEntry(className, returnType, nullable))
  }

  /**
   * Convenience method that registers both with Spark and with Comet in one call.
   *
   * @param spark
   *   The SparkSession
   * @param name
   *   The UDF name
   * @param className
   *   Fully-qualified CometUDF class name
   * @param sparkUdf
   *   The Spark UserDefinedFunction (for row-at-a-time fallback)
   * @param returnType
   *   The return DataType
   * @param nullable
   *   Whether the result may contain nulls
   */
  def register(
      spark: SparkSession,
      name: String,
      className: String,
      sparkUdf: org.apache.spark.sql.expressions.UserDefinedFunction,
      returnType: DataType,
      nullable: Boolean): Unit = {
    spark.udf.register(name, sparkUdf)
    registry.put(name, UdfEntry(className, returnType, nullable))
  }

  /**
   * Look up a registered CometUDF by its Spark UDF name.
   *
   * @return
   *   Some(UdfEntry) if registered, None otherwise
   */
  def get(name: String): Option[UdfEntry] = Option(registry.get(name))

  /**
   * Remove a previously registered UDF.
   */
  def remove(name: String): Unit = {
    registry.remove(name)
  }

  /**
   * Check whether a UDF name is registered.
   */
  def isRegistered(name: String): Boolean = registry.containsKey(name)

  // Visible for testing
  def size(): Int = registry.size()

  // Visible for testing
  def clear(): Unit = registry.clear()
}
