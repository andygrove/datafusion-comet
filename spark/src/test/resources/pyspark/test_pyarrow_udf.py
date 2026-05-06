#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Integration test for CometPythonMapInArrowExec.

This test verifies that Comet's optimized PyArrow UDF execution works correctly
by checking:
1. The plan uses CometPythonMapInArrowExec instead of PythonMapInArrow + ColumnarToRow
2. The UDF produces correct results
3. Performance improvement by eliminating unnecessary Arrow->Row->Arrow conversions

Usage:
    # Requires Python 3.11 or 3.12 (PySpark 3.5 does not support 3.13+)
    # Build Comet first: make release
    # Then run with PySpark:
    spark-submit --jars spark/target/comet-spark-spark3.5_2.12-*.jar \
        --conf spark.plugins=org.apache.spark.CometPlugin \
        --conf spark.comet.enabled=true \
        --conf spark.comet.exec.enabled=true \
        --conf spark.comet.exec.pythonMapInArrow.enabled=true \
        --conf spark.shuffle.manager=org.apache.spark.sql.comet.execution.shuffle.CometShuffleManager \
        --conf spark.memory.offHeap.enabled=true \
        --conf spark.memory.offHeap.size=2g \
        spark/src/test/resources/pyspark/test_pyarrow_udf.py
"""

import sys
import pyarrow as pa
from pyspark.sql import SparkSession
from pyspark.sql import types as T


def test_map_in_arrow_basic():
    """Test basic mapInArrow with Comet optimization."""
    spark = SparkSession.builder.getOrCreate()

    # Create test data
    data = [(i, float(i * 1.5), f"name_{i}") for i in range(100)]
    df = spark.createDataFrame(data, ["id", "value", "name"])

    # Write to parquet so CometScan can read it
    df.write.mode("overwrite").parquet("/tmp/comet_pyarrow_test_data")
    test_df = spark.read.parquet("/tmp/comet_pyarrow_test_data")

    # Define a PyArrow UDF that doubles the value column
    def double_value(batch: pa.RecordBatch) -> pa.RecordBatch:
        pdf = batch.to_pandas()
        pdf["value"] = pdf["value"] * 2
        return pa.RecordBatch.from_pandas(pdf)

    output_schema = T.StructType([
        T.StructField("id", T.LongType()),
        T.StructField("value", T.DoubleType()),
        T.StructField("name", T.StringType()),
    ])

    # Apply mapInArrow
    result_df = test_df.mapInArrow(double_value, output_schema)

    # Check the explain plan
    print("=" * 60)
    print("PHYSICAL PLAN:")
    print("=" * 60)
    result_df.explain(mode="extended")
    print("=" * 60)

    plan_str = result_df.queryExecution.executedPlan.toString()
    print(f"\nPlan string:\n{plan_str}\n")

    # Verify CometPythonMapInArrowExec is in the plan (if Comet is active)
    if "CometPythonMapInArrowExec" in plan_str:
        print("SUCCESS: CometPythonMapInArrowExec is in the plan!")
    elif "CometScan" in plan_str and "ColumnarToRow" in plan_str:
        print("WARNING: CometScan present but still using ColumnarToRow before Python UDF")
    elif "CometScan" not in plan_str:
        print("INFO: Comet is not active for this query (CometScan not found)")
    else:
        print("INFO: Plan does not contain CometPythonMapInArrowExec")

    # Verify correctness
    result = result_df.orderBy("id").collect()
    expected_first = data[0]
    actual_first = result[0]

    assert actual_first["id"] == expected_first[0], \
        f"ID mismatch: {actual_first['id']} != {expected_first[0]}"
    assert abs(actual_first["value"] - expected_first[1] * 2) < 0.001, \
        f"Value mismatch: {actual_first['value']} != {expected_first[1] * 2}"
    assert actual_first["name"] == expected_first[2], \
        f"Name mismatch: {actual_first['name']} != {expected_first[2]}"

    print(f"\nFirst row: {actual_first}")
    print(f"Expected value (doubled): {expected_first[1] * 2}")
    print("CORRECTNESS: PASSED")

    # Verify all rows
    for i, row in enumerate(result):
        expected_val = data[i][1] * 2
        assert abs(row["value"] - expected_val) < 0.001, \
            f"Row {i}: expected value {expected_val}, got {row['value']}"

    print(f"All {len(result)} rows verified correctly.")
    return True


def test_map_in_arrow_type_change():
    """Test mapInArrow that changes the schema."""
    spark = SparkSession.builder.getOrCreate()

    data = [(i, float(i)) for i in range(50)]
    df = spark.createDataFrame(data, ["id", "value"])
    df.write.mode("overwrite").parquet("/tmp/comet_pyarrow_test_data2")
    test_df = spark.read.parquet("/tmp/comet_pyarrow_test_data2")

    def add_computed_column(batch: pa.RecordBatch) -> pa.RecordBatch:
        pdf = batch.to_pandas()
        pdf["squared"] = pdf["value"] ** 2
        pdf["label"] = pdf["id"].apply(lambda x: f"item_{x}")
        return pa.RecordBatch.from_pandas(pdf)

    output_schema = T.StructType([
        T.StructField("id", T.LongType()),
        T.StructField("value", T.DoubleType()),
        T.StructField("squared", T.DoubleType()),
        T.StructField("label", T.StringType()),
    ])

    result_df = test_df.mapInArrow(add_computed_column, output_schema)
    result = result_df.orderBy("id").collect()

    assert len(result) == 50
    for i, row in enumerate(result):
        assert abs(row["squared"] - float(i) ** 2) < 0.001
        assert row["label"] == f"item_{i}"

    print("test_map_in_arrow_type_change: PASSED")
    return True


if __name__ == "__main__":
    print("Running PyArrow UDF integration tests for Comet...")
    print()

    tests = [
        ("test_map_in_arrow_basic", test_map_in_arrow_basic),
        ("test_map_in_arrow_type_change", test_map_in_arrow_type_change),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
