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

import org.junit.jupiter.api.Test;

import org.apache.comet.serde.Config.ConfigMap;
import org.apache.comet.serde.OperatorOuterClass.Operator;
import org.apache.comet.serde.OperatorOuterClass.Projection;

import static org.junit.jupiter.api.Assertions.*;

/** Verifies the datafusion-java module compiles correctly and protobuf classes are generated. */
class SessionContextTest {

  @Test
  void protobufClassesAreAccessible() {
    Operator op =
        Operator.newBuilder().setPlanId(1).setProjection(Projection.newBuilder().build()).build();

    byte[] serialized = op.toByteArray();
    assertNotNull(serialized);
    assertTrue(serialized.length > 0);
  }

  @Test
  void configMapSerialization() {
    ConfigMap config = ConfigMap.newBuilder().putEntries("key", "value").build();

    byte[] serialized = config.toByteArray();
    assertNotNull(serialized);
    assertTrue(serialized.length > 0);
  }

  @Test
  void sessionContextInstantiation() {
    try (SessionContext ctx = new SessionContext()) {
      assertNotNull(ctx);
    }
  }
}
