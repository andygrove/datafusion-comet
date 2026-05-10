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

package org.apache.datafusion.jni;

/**
 * JNI bridge to the native DataFusion execution engine. Declares native methods for creating,
 * executing, and releasing DataFusion query plans.
 *
 * <p>Native symbols for these methods will be provided in a future PR. Until then, calling these
 * methods will throw {@link UnsatisfiedLinkError}.
 */
public class DataFusionNative extends NativeLoader {

  /**
   * Create a native execution plan from a protobuf-serialized query plan.
   *
   * @param id unique identifier for this execution context
   * @param iterators input BatchIterator instances providing data to scan nodes
   * @param serializedPlan protobuf-serialized Operator message (the query plan)
   * @param serializedConfig protobuf-serialized ConfigMap message
   * @param partitionCount number of partitions
   * @param batchSize maximum number of rows per output batch
   * @param memoryPoolType type of memory pool ("greedy", "fair", or "unified")
   * @param memoryLimit maximum memory in bytes for this execution
   * @return opaque handle to the native execution context
   */
  public native long createPlan(
      long id,
      Object[] iterators,
      byte[] serializedPlan,
      byte[] serializedConfig,
      int partitionCount,
      int batchSize,
      String memoryPoolType,
      long memoryLimit);

  /**
   * Execute the plan and retrieve the next output batch.
   *
   * <p>The output batch is written to the provided Arrow C Data Interface addresses. Each call
   * returns the next batch until the stream is exhausted.
   *
   * @param planHandle opaque handle returned by {@link #createPlan}
   * @param arrayAddrs memory addresses of ArrowArray structures for output columns
   * @param schemaAddrs memory addresses of ArrowSchema structures for output columns
   * @return number of rows in the output batch, or -1 if the stream is exhausted
   */
  public native long executePlan(long planHandle, long[] arrayAddrs, long[] schemaAddrs);

  /**
   * Release the native execution context and free all associated resources.
   *
   * @param planHandle opaque handle returned by {@link #createPlan}
   */
  public native void releasePlan(long planHandle);
}
