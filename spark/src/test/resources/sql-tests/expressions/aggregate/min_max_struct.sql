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

-- min and max over a struct. Native comparison skips a field that is null on either side instead
-- of ordering it first as Spark does, so a struct is only Compatible when every field, at any
-- depth, is non-nullable. Parquet columns are always nullable, so the structs below are built
-- with COALESCE over a literal, which Spark knows is never null. The fields can still hold the
-- values the COALESCE substitutes for NULL.
-- ConfigMatrix: parquet.enable.dictionary=false,true

statement
CREATE TABLE test_mm_struct(
  grp string, ts bigint, a string, b string, d decimal(38,0), dt date, tstz timestamp,
  flag boolean, bin binary)
USING parquet

-- Group 'latest' is the max-by pattern: the largest ts wins and its payload comes along. Group
-- 'ties' has equal ts with different payloads, so the later fields decide. Group 'bounds' holds the
-- extreme bigint and decimal values and the earliest dates and timestamps Spark 3.x writes without
-- rebasing. Group 'cjk' holds strings whose UTF-8 order differs from their UTF-16 order, group
-- 'nulls' rows whose fields are all NULL, and group 'none' only NULLs.
statement
INSERT INTO test_mm_struct VALUES
  ('latest', 10, 'a1', 'b1', 1, DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', true, X'01'),
  ('latest', 30, 'a3', 'b3', 3, DATE'2020-01-03', TIMESTAMP'2020-01-03 00:00:00', false, X'03'),
  ('latest', 20, 'a2', 'b2', 2, DATE'2020-01-02', TIMESTAMP'2020-01-02 00:00:00', true, X'02'),
  ('ties', 5, 'x', 'p', 1, DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', true, X'01'),
  ('ties', 5, 'x', 'q', 1, DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', true, X'01'),
  ('ties', 5, 'w', 'z', 9, DATE'2021-01-01', TIMESTAMP'2021-01-01 00:00:00', false, X'FF'),
  ('ties', 5, 'x', 'q', 1, DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', true, X'00'),
  ('bounds', 9223372036854775807, 'max', 'max', 99999999999999999999999999999999999999,
    DATE'9999-12-31', TIMESTAMP'9999-12-31 23:59:59.999999', true, X'FF'),
  ('bounds', -9223372036854775808, 'min', 'min', -99999999999999999999999999999999999999,
    DATE'1900-01-02', TIMESTAMP'1900-01-02 00:00:00', false, X'00'),
  ('bounds', 0, '', '', 0, DATE'1970-01-01', TIMESTAMP'1970-01-01 00:00:00', false, X''),
  ('cjk', 1, 'ｱ', '😀', 1, NULL, NULL, NULL, NULL),
  ('cjk', 1, '😀', 'ｱ', 1, NULL, NULL, NULL, NULL),
  ('cjk', 1, '日本', '語', 1, NULL, NULL, NULL, NULL),
  ('cjk', 1, 'é', 'z', 1, NULL, NULL, NULL, NULL),
  ('nulls', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
  ('nulls', 7, NULL, 'n', NULL, NULL, NULL, NULL, NULL),
  ('none', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)

-- Every field is COALESCEd, so the optimizer cannot see a nullable field and the view keeps the
-- non-nullable struct type. A NULL ts becomes Long.MinValue, which orders before every real ts.
statement
CREATE OR REPLACE TEMPORARY VIEW v_mm_struct AS
SELECT grp,
  named_struct('ts', coalesce(ts, -9223372036854775808), 'a', coalesce(a, ''),
    'b', coalesce(b, '')) AS s,
  named_struct('ts', coalesce(ts, -9223372036854775808),
    'inner', named_struct('a', coalesce(a, ''), 'b', coalesce(b, ''))) AS nested,
  named_struct('d', coalesce(d, CAST(0 AS DECIMAL(38,0))),
    'dt', coalesce(dt, DATE'1970-01-01'),
    'tstz', coalesce(tstz, TIMESTAMP'1970-01-01 00:00:00'),
    'flag', coalesce(flag, false), 'bin', coalesce(bin, X'')) AS typed,
  CASE WHEN ts IS NOT NULL
    THEN named_struct('ts', coalesce(ts, 0L), 'a', coalesce(a, '')) END AS guarded
FROM test_mm_struct

-- Spark plans struct min/max as SortAggregateExec, as the struct buffer is not hash-friendly.
query
SELECT grp, max(s), min(s) FROM v_mm_struct GROUP BY grp ORDER BY grp

query
SELECT max(s), min(s) FROM v_mm_struct

-- The max-by pattern: pick the row with the largest ts and read its payload.
query
SELECT grp, max(s).a, max(s).b FROM v_mm_struct WHERE grp = 'latest' GROUP BY grp

-- A nested struct is ordered field by field, descending into the inner struct.
query
SELECT grp, max(nested), min(nested) FROM v_mm_struct GROUP BY grp ORDER BY grp

-- Boolean, decimal, date, timestamp and binary fields, including their extreme values.
query
SELECT grp, max(typed), min(typed) FROM v_mm_struct GROUP BY grp ORDER BY grp

-- A row whose whole struct is NULL is ignored; a group with only such rows returns NULL.
query
SELECT grp, max(guarded), min(guarded) FROM v_mm_struct GROUP BY grp ORDER BY grp

-- No rows at all returns NULL, with and without grouping.
query
SELECT max(s), min(s) FROM v_mm_struct WHERE grp = 'missing'

query
SELECT grp, max(s), min(s) FROM v_mm_struct WHERE grp = 'missing' GROUP BY grp

-- Mixed with other aggregates in the same operator, and with a filter clause.
query
SELECT grp, count(*), max(s), min(nested), max(s.ts), max(s) FILTER (WHERE s.ts > 5)
FROM v_mm_struct GROUP BY grp ORDER BY grp

-- collect_set makes Spark plan ObjectHashAggregateExec instead.
query
SELECT grp, max(s), sort_array(collect_set(s.a)) FROM v_mm_struct GROUP BY grp ORDER BY grp

-- count(DISTINCT) makes Spark plan four aggregates, so the struct buffer also goes through a
-- PartialMerge.
query
SELECT grp, count(DISTINCT s.a), max(s), min(nested) FROM v_mm_struct GROUP BY grp ORDER BY grp

-- A VALUES input keeps the partial aggregate on Spark, while the final one merges Spark's struct
-- buffers natively. The plan is not fully native, so only the answer is checked.
query spark_answer_only
SELECT k, max(named_struct('ts', t, 'a', a)), min(named_struct('ts', t, 'a', a))
FROM VALUES (1, 10L, 'x'), (1, 20L, 'y'), (2, 5L, 'b'), (2, 5L, 'a') AS v(k, t, a)
GROUP BY k ORDER BY k

-- Window frames stay on Spark.
query spark_answer_only
SELECT grp, max(s) OVER (PARTITION BY grp), min(s) OVER (PARTITION BY grp ORDER BY s)
FROM v_mm_struct ORDER BY grp, s

-- A nullable field is incompatible, so the aggregate falls back to Spark unless allowIncompatible
-- is set (see min_max_struct_incompat.sql).
query expect_fallback(nullable field)
SELECT grp, max(struct(ts, a)) FROM test_mm_struct GROUP BY grp ORDER BY grp

query expect_fallback(nullable field)
SELECT grp, min(named_struct('ts', coalesce(ts, 0L), 'inner', named_struct('a', a))) FROM test_mm_struct
GROUP BY grp ORDER BY grp

-- A float or double field orders NaN differently from Spark, so it is unsupported at any depth.
query expect_fallback(Unsupported data type in struct)
SELECT grp, max(named_struct('ts', coalesce(ts, 0L), 'f', coalesce(CAST(ts AS DOUBLE), 0.0D)))
FROM test_mm_struct GROUP BY grp ORDER BY grp

query expect_fallback(Unsupported data type in struct)
SELECT grp, min(named_struct('ts', coalesce(ts, 0L),
  'inner', named_struct('f', coalesce(CAST(ts AS FLOAT), CAST(0 AS FLOAT)))))
FROM test_mm_struct GROUP BY grp ORDER BY grp

-- Arrays, maps and timestamp without time zone inside a struct are unsupported.
query expect_fallback(Unsupported data type in struct)
SELECT grp, max(named_struct('ts', coalesce(ts, 0L), 'arr', array(1, 2))) FROM test_mm_struct
GROUP BY grp ORDER BY grp

query expect_fallback(Unsupported data type in struct)
SELECT grp, max(named_struct('ts', coalesce(ts, 0L), 'ntz', CAST(tstz AS TIMESTAMP_NTZ)))
FROM test_mm_struct GROUP BY grp ORDER BY grp
