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

package org.apache.datafusion;

import java.util.Collections;
import java.util.Map;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.datafusion.jni.DataFusionNative;

import org.apache.comet.serde.Config.ConfigMap;

/**
 * Entry point for executing DataFusion query plans from Java. A SessionContext manages the
 * lifecycle of native execution resources.
 *
 * <p>Usage:
 *
 * <pre>{@code
 * try (SessionContext ctx = new SessionContext()) {
 *     byte[] plan = buildProtobufPlan();
 *     try (RecordBatchStream stream = ctx.executePlan(plan, 4)) {
 *         while (stream.hasNext()) {
 *             VectorSchemaRoot batch = stream.next();
 *             // process batch
 *             batch.close();
 *         }
 *     }
 * }
 * }</pre>
 */
public class SessionContext implements AutoCloseable {

  private static final String DEFAULT_MEMORY_POOL_TYPE = "greedy";
  private static final long DEFAULT_MEMORY_LIMIT = Long.MAX_VALUE;
  private static final int DEFAULT_BATCH_SIZE = 8192;

  private final DataFusionNative nativeApi;
  private final BufferAllocator allocator;
  private final Map<String, String> config;
  private final boolean ownsAllocator;

  private long nextPlanId = 0;

  public SessionContext() {
    this(Collections.emptyMap());
  }

  public SessionContext(Map<String, String> config) {
    this(config, new RootAllocator(Long.MAX_VALUE), true);
  }

  public SessionContext(Map<String, String> config, BufferAllocator allocator) {
    this(config, allocator, false);
  }

  private SessionContext(
      Map<String, String> config, BufferAllocator allocator, boolean ownsAllocator) {
    this.nativeApi = new DataFusionNative();
    this.allocator = allocator;
    this.config = config;
    this.ownsAllocator = ownsAllocator;
  }

  /**
   * Execute a DataFusion plan and return a stream of result batches.
   *
   * @param serializedPlan protobuf-serialized Operator message representing the query plan
   * @param numOutputColumns number of columns in the output schema
   * @param inputs BatchIterator instances providing input data to scan nodes in the plan
   * @return a stream of Arrow record batches
   */
  public RecordBatchStream executePlan(
      byte[] serializedPlan, int numOutputColumns, BatchIterator... inputs) {
    return executePlan(serializedPlan, numOutputColumns, 1, DEFAULT_BATCH_SIZE, inputs);
  }

  /**
   * Execute a DataFusion plan with full control over execution parameters.
   *
   * @param serializedPlan protobuf-serialized Operator message representing the query plan
   * @param numOutputColumns number of columns in the output schema
   * @param partitionCount number of partitions
   * @param batchSize maximum rows per output batch
   * @param inputs BatchIterator instances providing input data to scan nodes in the plan
   * @return a stream of Arrow record batches
   */
  public RecordBatchStream executePlan(
      byte[] serializedPlan,
      int numOutputColumns,
      int partitionCount,
      int batchSize,
      BatchIterator... inputs) {

    byte[] serializedConfig = serializeConfig();
    Object[] iterators = inputs != null ? inputs : new Object[0];

    long planHandle =
        nativeApi.createPlan(
            nextPlanId++,
            iterators,
            serializedPlan,
            serializedConfig,
            partitionCount,
            batchSize,
            DEFAULT_MEMORY_POOL_TYPE,
            DEFAULT_MEMORY_LIMIT);

    return new RecordBatchStream(nativeApi, planHandle, allocator, numOutputColumns);
  }

  private byte[] serializeConfig() {
    ConfigMap.Builder builder = ConfigMap.newBuilder();
    builder.putAllEntries(config);
    return builder.build().toByteArray();
  }

  @Override
  public void close() {
    if (ownsAllocator) {
      allocator.close();
    }
  }
}
