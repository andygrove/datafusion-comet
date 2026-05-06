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
Pytest-driven integration tests for Comet's PyArrow UDF acceleration.

Each test runs against two execution paths:
  - "accelerated": spark.comet.exec.pythonMapInArrow.enabled=true
                   (plan should contain CometPythonMapInArrow and no ColumnarToRow)
  - "fallback":    spark.comet.exec.pythonMapInArrow.enabled=false
                   (plan should contain vanilla PythonMapInArrow)

Usage:
    # Build Comet first:
    make release

    # Then either let the test discover the jar from spark/target, or pass it
    # explicitly via COMET_JAR:
    export COMET_JAR=$PWD/spark/target/comet-spark-spark3.5_2.12-0.16.0-SNAPSHOT.jar

    pip install pyspark==3.5.8 pyarrow pandas pytest
    pytest -v spark/src/test/resources/pyspark/test_pyarrow_udf.py
"""

import datetime as dt
import glob
import os
from decimal import Decimal

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession, types as T


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)


def _resolve_comet_jar() -> str:
    explicit = os.environ.get("COMET_JAR")
    if explicit:
        if any(ch in explicit for ch in "*?["):
            matches = sorted(glob.glob(explicit))
            if not matches:
                raise FileNotFoundError(
                    f"COMET_JAR pattern matched nothing: {explicit}"
                )
            return matches[-1]
        return explicit

    # Pick the jar that matches the installed pyspark major.minor version. The
    # Comet jars are published per Spark version (e.g., comet-spark-spark3.5_2.12-*.jar);
    # using the wrong one yields ClassNotFoundException on Scala stdlib classes.
    import pyspark

    major_minor = ".".join(pyspark.__version__.split(".")[:2])
    spark_tag = f"spark{major_minor}"
    scala_tag = "_2.12" if major_minor.startswith("3.") else "_2.13"
    pattern = os.path.join(
        REPO_ROOT,
        f"spark/target/comet-spark-{spark_tag}{scala_tag}-*-SNAPSHOT.jar",
    )
    candidates = [
        m
        for m in sorted(glob.glob(pattern))
        if "sources" not in os.path.basename(m) and "tests" not in os.path.basename(m)
    ]
    if not candidates:
        raise FileNotFoundError(
            "Comet jar not found. Set COMET_JAR or run `make release`. "
            f"Looked under {pattern}."
        )
    return candidates[-1]


@pytest.fixture(scope="session")
def spark():
    jar = _resolve_comet_jar()
    # PYSPARK_SUBMIT_ARGS is consumed when pyspark launches its JVM. Setting
    # --jars puts the Comet jar on both driver and executor classpaths so the
    # CometPlugin can be loaded.
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        f"--jars {jar} --driver-class-path {jar} pyspark-shell"
    )
    session = (
        SparkSession.builder.master("local[2]")
        .appName("comet-pyarrow-udf-tests")
        .config("spark.plugins", "org.apache.spark.CometPlugin")
        .config("spark.comet.enabled", "true")
        .config("spark.comet.exec.enabled", "true")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .getOrCreate()
    )
    try:
        yield session
    finally:
        session.stop()


@pytest.fixture(params=[True, False], ids=["accelerated", "fallback"])
def accelerated(request, spark) -> bool:
    spark.conf.set(
        "spark.comet.exec.pythonMapInArrow.enabled",
        "true" if request.param else "false",
    )
    return request.param


def _executed_plan(df) -> str:
    return df._jdf.queryExecution().executedPlan().toString()


def _assert_plan_matches_mode(
    plan: str, accelerated: bool, vanilla_node: str = "PythonMapInArrow"
) -> None:
    if accelerated:
        assert "CometPythonMapInArrow" in plan, (
            f"expected CometPythonMapInArrow in accelerated plan, got:\n{plan}"
        )
        assert "ColumnarToRow" not in plan, (
            f"unexpected ColumnarToRow in accelerated plan:\n{plan}"
        )
    else:
        assert "CometPythonMapInArrow" not in plan, (
            f"unexpected CometPythonMapInArrow in fallback plan:\n{plan}"
        )
        assert vanilla_node in plan, (
            f"expected {vanilla_node} in fallback plan, got:\n{plan}"
        )


def test_map_in_arrow_doubles_value(spark, tmp_path, accelerated):
    data = [(i, float(i * 1.5), f"name_{i}") for i in range(100)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(data, ["id", "value", "name"]).write.parquet(src)

    def double_value(iterator):
        for batch in iterator:
            pdf = batch.to_pandas()
            pdf["value"] = pdf["value"] * 2
            yield pa.RecordBatch.from_pandas(pdf)

    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
            T.StructField("name", T.StringType()),
        ]
    )
    result_df = spark.read.parquet(src).mapInArrow(double_value, schema)

    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    rows = result_df.orderBy("id").collect()
    assert len(rows) == len(data)
    for row, original in zip(rows, data):
        assert row["id"] == original[0]
        assert abs(row["value"] - original[1] * 2) < 1e-6
        assert row["name"] == original[2]


def test_map_in_arrow_changes_schema(spark, tmp_path, accelerated):
    data = [(i, float(i)) for i in range(50)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(data, ["id", "value"]).write.parquet(src)

    def add_computed_column(iterator):
        for batch in iterator:
            pdf = batch.to_pandas()
            pdf["squared"] = pdf["value"] ** 2
            pdf["label"] = pdf["id"].apply(lambda x: f"item_{x}")
            yield pa.RecordBatch.from_pandas(pdf)

    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
            T.StructField("squared", T.DoubleType()),
            T.StructField("label", T.StringType()),
        ]
    )
    result_df = spark.read.parquet(src).mapInArrow(add_computed_column, schema)

    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    rows = result_df.orderBy("id").collect()
    assert len(rows) == 50
    for i, row in enumerate(rows):
        assert abs(row["squared"] - float(i) ** 2) < 1e-6
        assert row["label"] == f"item_{i}"


def test_map_in_pandas_doubles_value(spark, tmp_path, accelerated):
    data = [(i, float(i * 1.5)) for i in range(100)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(data, ["id", "value"]).write.parquet(src)

    def double_value(iterator):
        for pdf in iterator:
            pdf = pdf.copy()
            pdf["value"] = pdf["value"] * 2
            yield pdf

    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
        ]
    )
    result_df = spark.read.parquet(src).mapInPandas(double_value, schema)

    _assert_plan_matches_mode(
        _executed_plan(result_df), accelerated, vanilla_node="MapInPandas"
    )

    rows = result_df.orderBy("id").collect()
    assert len(rows) == len(data)
    for row, original in zip(rows, data):
        assert row["id"] == original[0]
        assert abs(row["value"] - original[1] * 2) < 1e-6


def test_map_in_pandas_changes_schema(spark, tmp_path, accelerated):
    data = [(i, float(i)) for i in range(50)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(data, ["id", "value"]).write.parquet(src)

    def add_squared(iterator):
        for pdf in iterator:
            pdf = pdf.copy()
            pdf["squared"] = pdf["value"] ** 2
            yield pdf

    schema = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
            T.StructField("squared", T.DoubleType()),
        ]
    )
    result_df = spark.read.parquet(src).mapInPandas(add_squared, schema)

    _assert_plan_matches_mode(
        _executed_plan(result_df), accelerated, vanilla_node="MapInPandas"
    )

    rows = result_df.orderBy("id").collect()
    assert len(rows) == 50
    for i, row in enumerate(rows):
        assert abs(row["squared"] - float(i) ** 2) < 1e-6


def test_map_in_arrow_preserves_nulls(spark, tmp_path, accelerated):
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("name", T.StringType()),
        ]
    )
    rows = [
        (1, "a"),
        (2, None),
        (None, "c"),
        (None, None),
        (5, "e"),
    ]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(rows, schema_in).write.parquet(src)

    def passthrough(iterator):
        # Pure Arrow passthrough so nulls survive without a pandas roundtrip
        # (pandas would coerce null longs to NaN floats).
        for batch in iterator:
            yield batch

    result_df = spark.read.parquet(src).mapInArrow(passthrough, schema_in)
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    out = {(r["id"], r["name"]) for r in result_df.collect()}
    assert out == set(rows)


def test_map_in_arrow_empty_input(spark, tmp_path, accelerated):
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
        ]
    )
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame([(1, 1.0), (2, 2.0)], schema_in).write.parquet(src)

    def passthrough(iterator):
        for batch in iterator:
            yield batch

    # Filter all rows out so the operator sees an empty stream from CometScan.
    result_df = (
        spark.read.parquet(src).where("id < 0").mapInArrow(passthrough, schema_in)
    )
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    assert result_df.count() == 0


def test_map_in_arrow_python_exception_propagates(spark, tmp_path, accelerated):
    schema_in = T.StructType([T.StructField("id", T.LongType())])
    data = [(i,) for i in range(10)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(data, schema_in).write.parquet(src)

    sentinel = "boom-from-pyarrow-udf"

    def boom(iterator):
        for _batch in iterator:
            raise ValueError(sentinel)
        # Unreachable, but mapInArrow requires the callable to be a generator.
        yield  # pragma: no cover

    result_df = spark.read.parquet(src).mapInArrow(boom, schema_in)
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    with pytest.raises(Exception) as exc_info:
        result_df.collect()
    assert sentinel in str(exc_info.value), (
        f"expected sentinel {sentinel!r} in exception, got: {exc_info.value}"
    )


def test_map_in_arrow_decimal_type(spark, tmp_path, accelerated):
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("amount", T.DecimalType(18, 6)),
        ]
    )
    rows = [
        (1, Decimal("123.456789")),
        (2, Decimal("0.000001")),
        (3, Decimal("-99999999.999999")),
        (4, None),
    ]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(rows, schema_in).write.parquet(src)

    def passthrough(iterator):
        for batch in iterator:
            yield batch

    result_df = spark.read.parquet(src).mapInArrow(passthrough, schema_in)
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    out = {(r["id"], r["amount"]) for r in result_df.collect()}
    assert out == set(rows)


def test_map_in_arrow_date_and_timestamp(spark, tmp_path, accelerated):
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("d", T.DateType()),
            T.StructField("ts", T.TimestampType()),
        ]
    )
    rows = [
        (1, dt.date(2024, 1, 1), dt.datetime(2024, 1, 1, 12, 30, 45)),
        (2, dt.date(1999, 12, 31), dt.datetime(2000, 6, 15, 0, 0, 0)),
        (3, None, None),
    ]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(rows, schema_in).write.parquet(src)

    def passthrough(iterator):
        for batch in iterator:
            yield batch

    result_df = spark.read.parquet(src).mapInArrow(passthrough, schema_in)
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    out = {(r["id"], r["d"], r["ts"]) for r in result_df.collect()}
    assert out == set(rows)


def test_map_in_arrow_array_and_struct(spark, tmp_path, accelerated):
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("nums", T.ArrayType(T.IntegerType())),
            T.StructField(
                "addr",
                T.StructType(
                    [
                        T.StructField("city", T.StringType()),
                        T.StructField("zip", T.IntegerType()),
                    ]
                ),
            ),
        ]
    )
    rows = [
        (1, [1, 2, 3], ("Berlin", 10115)),
        (2, [], ("NYC", 10001)),
        (3, None, None),
        (4, [None, 5], ("Tokyo", None)),
    ]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(rows, schema_in).write.parquet(src)

    def passthrough(iterator):
        for batch in iterator:
            yield batch

    result_df = spark.read.parquet(src).mapInArrow(passthrough, schema_in)
    _assert_plan_matches_mode(_executed_plan(result_df), accelerated)

    def _normalize(row):
        nums = tuple(row["nums"]) if row["nums"] is not None else None
        addr = row["addr"]
        addr_tuple = (addr["city"], addr["zip"]) if addr is not None else None
        return (row["id"], nums, addr_tuple)

    out = {_normalize(r) for r in result_df.collect()}
    expected = {
        (r[0], tuple(r[1]) if r[1] is not None else None, r[2]) for r in rows
    }
    assert out == expected


def test_map_in_arrow_after_shuffle(spark, tmp_path, accelerated):
    """
    Verifies correctness when a shuffle sits between the Comet scan and the
    Python UDF. Without `spark.shuffle.manager` configured at session startup
    the shuffle stays a vanilla `Exchange`, which is not columnar, so the
    optimization does not fire across it today. This test does not assert on
    the plan; it only ensures the path produces correct results in both modes
    so a future change that wires Comet shuffle into the optimization does
    not silently break correctness.
    """
    schema_in = T.StructType(
        [
            T.StructField("id", T.LongType()),
            T.StructField("value", T.DoubleType()),
        ]
    )
    rows = [(i, float(i)) for i in range(50)]
    src = str(tmp_path / "src.parquet")
    spark.createDataFrame(rows, schema_in).write.parquet(src)

    def passthrough(iterator):
        for batch in iterator:
            yield batch

    result_df = (
        spark.read.parquet(src)
        .repartition(4, "id")
        .mapInArrow(passthrough, schema_in)
    )

    out = sorted((r["id"], r["value"]) for r in result_df.collect())
    assert out == sorted(rows)
