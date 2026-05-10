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

/**
 * Interface for providing input record batches to native DataFusion execution. Native code calls
 * back into this interface via JNI to pull input data.
 *
 * <p>Implementations export Arrow columnar data using the Arrow C Data Interface, passing memory
 * addresses of ArrowArray and ArrowSchema structures.
 */
public interface BatchIterator {

  /**
   * Prepare the next batch for consumption.
   *
   * @return the number of rows in the next batch, or -1 if there are no more batches
   */
  int hasNext();

  /**
   * Export the current batch into the provided Arrow C Data Interface addresses.
   *
   * @param arrayAddrs memory addresses of ArrowArray structures (one per column)
   * @param schemaAddrs memory addresses of ArrowSchema structures (one per column)
   * @return the number of rows exported, or -1 if no batch is available
   */
  int next(long[] arrayAddrs, long[] schemaAddrs);
}
