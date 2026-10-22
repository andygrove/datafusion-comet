-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
--   http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

-- min and max over a struct with a nullable field are incompatible: native comparison skips a field
-- that is null on either side, while Spark orders NULL before every other value. They run natively
-- when allowIncompatible is set. Every query below is one where the two orderings agree, because
-- the first field that differs between two rows is non-NULL on both, or the NULL fields are equal.
-- Config: spark.comet.expression.Max.allowIncompatible=true
-- Config: spark.comet.expression.Min.allowIncompatible=true
-- ConfigMatrix: parquet.enable.dictionary=false,true

statement
CREATE TABLE test_mm_struct_nullable(grp string, ts bigint, a string, b string) USING parquet

statement
INSERT INTO test_mm_struct_nullable VALUES
  ('latest', 10, 'a1', 'b1'), ('latest', 30, 'a3', NULL), ('latest', 20, NULL, 'b2'),
  ('equal', 5, NULL, 'x'), ('equal', 5, NULL, 'x'),
  ('nullpayload', 5, 'a', NULL), ('nullpayload', 4, NULL, 'b'),
  ('allnull', NULL, NULL, NULL), ('allnull', NULL, NULL, NULL)

-- The max-by pattern, whose ordering field is never NULL for the rows that differ.
query
SELECT grp, max(struct(ts, a, b)), min(struct(ts, a, b)) FROM test_mm_struct_nullable
WHERE grp IN ('latest', 'equal', 'nullpayload') GROUP BY grp ORDER BY grp

query
SELECT max(struct(ts, a, b)), min(struct(ts, a, b)) FROM test_mm_struct_nullable
WHERE grp IN ('latest', 'nullpayload')

-- A group of only NULL fields is a struct of NULLs, not a NULL struct.
query
SELECT grp, max(struct(ts, a, b)), min(struct(ts, a, b)) FROM test_mm_struct_nullable
WHERE grp = 'allnull' GROUP BY grp

-- Nested struct whose differing field is non-NULL.
query
SELECT grp, max(struct(ts, struct(a, b))) FROM test_mm_struct_nullable
WHERE grp IN ('latest', 'nullpayload') GROUP BY grp ORDER BY grp

-- A float field is still unsupported, as native comparison does not order NaN as Spark does.
query expect_fallback(Unsupported data type in struct)
SELECT grp, max(struct(ts, CAST(ts AS DOUBLE))) FROM test_mm_struct_nullable GROUP BY grp ORDER BY grp
