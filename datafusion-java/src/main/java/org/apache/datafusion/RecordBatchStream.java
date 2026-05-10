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

import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.CDataDictionaryProvider;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.datafusion.jni.DataFusionNative;

/**
 * An iterator over Arrow record batches produced by executing a DataFusion plan. Each call to
 * {@link #next()} returns the next batch of results.
 *
 * <p>This class manages Arrow C Data Interface memory for importing results from native code. It
 * must be closed when no longer needed to release native resources.
 */
public class RecordBatchStream implements Iterator<VectorSchemaRoot>, AutoCloseable {

  private final DataFusionNative nativeApi;
  private final long planHandle;
  private final BufferAllocator allocator;
  private final int numColumns;
  private final long[] arrayAddrs;
  private final long[] schemaAddrs;
  private final ArrowArray[] arrowArrays;
  private final ArrowSchema[] arrowSchemas;

  private boolean finished = false;
  private boolean closed = false;
  private Long pendingRowCount = null;

  RecordBatchStream(
      DataFusionNative nativeApi, long planHandle, BufferAllocator allocator, int numColumns) {
    this.nativeApi = nativeApi;
    this.planHandle = planHandle;
    this.allocator = allocator;
    this.numColumns = numColumns;

    this.arrayAddrs = new long[numColumns];
    this.schemaAddrs = new long[numColumns];
    this.arrowArrays = new ArrowArray[numColumns];
    this.arrowSchemas = new ArrowSchema[numColumns];

    for (int i = 0; i < numColumns; i++) {
      arrowArrays[i] = ArrowArray.allocateNew(allocator);
      arrowSchemas[i] = ArrowSchema.allocateNew(allocator);
      arrayAddrs[i] = arrowArrays[i].memoryAddress();
      schemaAddrs[i] = arrowSchemas[i].memoryAddress();
    }
  }

  @Override
  public boolean hasNext() {
    if (closed || finished) {
      return false;
    }
    if (pendingRowCount != null) {
      return true;
    }

    long rowCount = nativeApi.executePlan(planHandle, arrayAddrs, schemaAddrs);
    if (rowCount < 0) {
      finished = true;
      return false;
    }
    pendingRowCount = rowCount;
    return true;
  }

  @Override
  public VectorSchemaRoot next() {
    if (!hasNext()) {
      throw new NoSuchElementException("No more batches");
    }

    CDataDictionaryProvider dictProvider = new CDataDictionaryProvider();
    try {
      FieldVector[] vectors = new FieldVector[numColumns];
      for (int i = 0; i < numColumns; i++) {
        vectors[i] = Data.importVector(allocator, arrowArrays[i], arrowSchemas[i], dictProvider);
        // Re-allocate C Data structs for next call
        arrowArrays[i] = ArrowArray.allocateNew(allocator);
        arrowSchemas[i] = ArrowSchema.allocateNew(allocator);
        arrayAddrs[i] = arrowArrays[i].memoryAddress();
        schemaAddrs[i] = arrowSchemas[i].memoryAddress();
      }

      Field[] fields = new Field[numColumns];
      for (int i = 0; i < numColumns; i++) {
        fields[i] = vectors[i].getField();
      }

      VectorSchemaRoot root =
          new VectorSchemaRoot(
              new Schema(java.util.Arrays.asList(fields)),
              java.util.Arrays.asList(vectors),
              pendingRowCount.intValue());
      pendingRowCount = null;
      return root;
    } finally {
      dictProvider.close();
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;

    for (int i = 0; i < numColumns; i++) {
      if (arrowArrays[i] != null) {
        arrowArrays[i].close();
      }
      if (arrowSchemas[i] != null) {
        arrowSchemas[i].close();
      }
    }
  }
}
