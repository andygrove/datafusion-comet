<!---
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# PyArrow UDF Acceleration

Comet can accelerate Python UDFs that use PyArrow-backed batch processing, such as `mapInArrow` and `mapInPandas`.
These APIs are commonly used for ML inference, feature engineering, and data transformation workloads.

## Background

Spark's `mapInArrow` and `mapInPandas` APIs allow users to apply Python functions that operate on Arrow
RecordBatches or Pandas DataFrames. Under the hood, Spark communicates with the Python worker process
using the Arrow IPC format.

Without Comet, the execution path for these UDFs involves unnecessary data conversions:

1. Comet reads data in Arrow columnar format (via CometScan)
2. Spark inserts a ColumnarToRow transition (converts Arrow to UnsafeRow)
3. The Python runner converts those rows back to Arrow to send to Python
4. Python executes the UDF on Arrow batches
5. Results are returned as Arrow and then converted back to rows

Steps 2 and 3 are redundant since the data starts and ends in Arrow format.

## How Comet Optimizes This

When enabled, Comet detects `PythonMapInArrowExec` and `MapInPandasExec` operators in the physical plan
and replaces them with `CometPythonMapInArrowExec`, which:

- Reads Arrow columnar batches directly from the upstream Comet operator
- Feeds them to the Python runner without the expensive UnsafeProjection copy
- Keeps the Python output in columnar format for downstream operators

This eliminates the ColumnarToRow transition and the output row conversion, reducing CPU overhead
and memory allocations.

## Configuration

The optimization is experimental and disabled by default. Enable it with:

```
spark.comet.exec.pythonMapInArrow.enabled=true
```

The default is `false` while the feature stabilizes.

## Supported APIs

| PySpark API                      | Spark Plan Node             | Supported |
| -------------------------------- | --------------------------- | --------- |
| `df.mapInArrow(func, schema)`    | `PythonMapInArrowExec`      | Yes       |
| `df.mapInPandas(func, schema)`   | `MapInPandasExec`           | Yes       |
| `@pandas_udf` (scalar)           | `ArrowEvalPythonExec`       | Not yet   |
| `df.applyInPandas(func, schema)` | `FlatMapGroupsInPandasExec` | Not yet   |

## Example

```python
import pyarrow as pa
from pyspark.sql import SparkSession, types as T

spark = SparkSession.builder \
    .config("spark.plugins", "org.apache.spark.CometPlugin") \
    .config("spark.comet.enabled", "true") \
    .config("spark.comet.exec.enabled", "true") \
    .config("spark.comet.exec.pythonMapInArrow.enabled", "true") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

df = spark.read.parquet("data.parquet")

def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
    # Your transformation logic here
    table = batch.to_pandas()
    table["new_col"] = table["value"] * 2
    return pa.RecordBatch.from_pandas(table)

output_schema = T.StructType([
    T.StructField("value", T.DoubleType()),
    T.StructField("new_col", T.DoubleType()),
])

result = df.mapInArrow(transform, output_schema)
```

## Verifying the Optimization

Use `explain()` to verify that `CometPythonMapInArrowExec` appears in your plan:

```python
result.explain(mode="extended")
```

You should see:

```
CometPythonMapInArrowExec ...
+- CometNativeExec ...
   +- CometScan ...
```

Instead of the unoptimized plan:

```
PythonMapInArrow ...
+- ColumnarToRow
   +- CometNativeExec ...
      +- CometScan ...
```

## Limitations

- The optimization currently applies only to `mapInArrow` and `mapInPandas`. Scalar pandas UDFs
  (`@pandas_udf`) and grouped operations (`applyInPandas`) are not yet supported.
- The internal row-to-Arrow conversion inside the Python runner is still present in this version.
  A future optimization will write Arrow batches directly to the Python IPC stream, achieving
  near zero-copy data transfer.
