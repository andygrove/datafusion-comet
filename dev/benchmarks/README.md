<!--
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

# Comet Benchmarking Scripts

This directory contains scripts used for generating benchmark results that are published in this repository and in
the Comet documentation.

For full instructions on running these benchmarks on an EC2 instance, see the [Comet Benchmarking on EC2 Guide].

[Comet Benchmarking on EC2 Guide]: https://datafusion.apache.org/comet/contributor-guide/benchmarking_aws_ec2.html

## Example usage

Set Spark environment variables:

```shell
export SPARK_HOME=/opt/spark-3.5.3-bin-hadoop3/
export SPARK_MASTER=spark://yourhostname:7077
```

Set path to queries and data:

```shell
export TPCH_QUERIES=/mnt/bigdata/tpch/queries/
export TPCH_DATA=/mnt/bigdata/tpch/sf100/
```

Run Spark benchmark:

```shell
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
sudo ./drop-caches.sh
./spark-tpch.sh
```

Run Comet benchmark:

```shell
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export COMET_JAR=/opt/comet/comet-spark-spark3.5_2.12-0.10.0.jar
sudo ./drop-caches.sh
./comet-tpch.sh
```

Run Gluten benchmark:

```shell
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export GLUTEN_JAR=/opt/gluten/gluten-velox-bundle-spark3.5_2.12-linux_amd64-1.4.0.jar
sudo ./drop-caches.sh
./gluten-tpch.sh
```

Generating charts:

```shell
python3 generate-comparison.py --benchmark tpch --labels "Spark 3.5.3" "Comet 0.9.0" "Gluten 1.4.0" --title "TPC-H @ 100 GB (single executor, 8 cores, local Parquet files)" spark-tpch-1752338506381.json comet-tpch-1752337818039.json gluten-tpch-1752337474344.json
```

## Iceberg Benchmarking

Comet includes native Iceberg support via iceberg-rust integration. This enables benchmarking TPC-H queries
against Iceberg tables with native scan acceleration.

### Prerequisites

Download the Iceberg Spark runtime JAR (required for running the benchmark):

```shell
wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.8.1/iceberg-spark-runtime-3.5_2.12-1.8.1.jar
export ICEBERG_JAR=/path/to/iceberg-spark-runtime-3.5_2.12-1.8.1.jar
```

Note: Table creation uses `--packages` which auto-downloads the dependency.

### Create Iceberg TPC-H tables

Convert existing Parquet TPC-H data to Iceberg format:

```shell
export ICEBERG_WAREHOUSE=/mnt/bigdata/iceberg-warehouse
export ICEBERG_CATALOG=${ICEBERG_CATALOG:-local}

$SPARK_HOME/bin/spark-submit \
    --master $SPARK_MASTER \
    --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.8.1 \
    --conf spark.driver.memory=8G \
    --conf spark.executor.instances=1 \
    --conf spark.executor.cores=8 \
    --conf spark.cores.max=8 \
    --conf spark.executor.memory=16g \
    --conf spark.sql.catalog.${ICEBERG_CATALOG}=org.apache.iceberg.spark.SparkCatalog \
    --conf spark.sql.catalog.${ICEBERG_CATALOG}.type=hadoop \
    --conf spark.sql.catalog.${ICEBERG_CATALOG}.warehouse=$ICEBERG_WAREHOUSE \
    create-iceberg-tpch.py \
    --parquet-path $TPCH_DATA \
    --catalog $ICEBERG_CATALOG \
    --database tpch
```

### Run Iceberg benchmark

```shell
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export COMET_JAR=/opt/comet/comet-spark-spark3.5_2.12-0.10.0.jar
export ICEBERG_JAR=/path/to/iceberg-spark-runtime-3.5_2.12-1.8.1.jar
export ICEBERG_WAREHOUSE=/mnt/bigdata/iceberg-warehouse
export TPCH_QUERIES=/mnt/bigdata/tpch/queries/
sudo ./drop-caches.sh
./comet-tpch-iceberg.sh
```

The benchmark uses `spark.comet.scan.icebergNative.enabled=true` to enable Comet's native iceberg-rust
integration. Verify native scanning is active by checking for `CometIcebergNativeScanExec` in the
physical plan output.

### Iceberg-specific options

| Environment Variable | Default    | Description                         |
| -------------------- | ---------- | ----------------------------------- |
| `ICEBERG_CATALOG`    | `local`    | Iceberg catalog name                |
| `ICEBERG_DATABASE`   | `tpch`     | Database containing TPC-H tables    |
| `ICEBERG_WAREHOUSE`  | (required) | Path to Iceberg warehouse directory |

### Comparing Parquet vs Iceberg performance

Run both benchmarks and compare:

```shell
python3 generate-comparison.py --benchmark tpch \
    --labels "Comet (Parquet)" "Comet (Iceberg)" \
    --title "TPC-H @ 100 GB: Parquet vs Iceberg" \
    comet-tpch-*.json comet-iceberg-tpch-*.json
```

## Docker Benchmarking

A `Dockerfile` is provided for running benchmarks in a container with controllable CPU and memory
constraints. This is useful for reproducible benchmarking without dedicated hardware.

### Prerequisites

Before building the Docker image, you need the following on the host:

- TPC-H or TPC-DS data in Parquet format (generate with [tpchgen-cli])
- TPC-H or TPC-DS SQL query files
- A pre-built Comet JAR (e.g. from `make release`)

[tpchgen-cli]: https://github.com/clarkzjw/tpchgen-rs

### Build the image

```shell
cd dev/benchmarks
docker build -t comet-bench .
```

### Volume mounts

The container expects data to be mounted at these paths:

| Mount Point  | Description                                        |
| ------------ | -------------------------------------------------- |
| `/data`      | TPC-H or TPC-DS data directory (Parquet files)     |
| `/queries`   | SQL query files (`q1.sql` through `q22.sql`, etc.) |
| `/jars`      | Directory containing the Comet JAR                 |
| `/results`   | Output directory for benchmark result JSON files   |

### Run TPC-H benchmark

```shell
docker run --rm \
    -v /path/to/tpch/data:/data:ro \
    -v /path/to/tpch/queries:/queries:ro \
    -v /path/to/comet-jar-dir:/jars:ro \
    -v /path/to/results:/results \
    -e COMET_JAR=/jars/comet-spark-spark3.5_2.12-0.10.0.jar \
    comet-bench \
    bash -c "./comet-tpch.sh"
```

For TPC-DS, use `comet-tpcds.sh` and mount TPC-DS data and queries instead.

### Running with CPU and memory constraints

Docker's `--cpus` and `--memory` flags let you simulate different hardware configurations
for reproducible benchmarks.

**Limit to 8 CPUs and 32 GB of memory:**

```shell
docker run --rm \
    --cpus=8 \
    --memory=32g \
    -v /path/to/tpch/data:/data:ro \
    -v /path/to/tpch/queries:/queries:ro \
    -v /path/to/comet-jar-dir:/jars:ro \
    -v /path/to/results:/results \
    -e COMET_JAR=/jars/comet-spark-spark3.5_2.12-0.10.0.jar \
    comet-bench \
    bash -c "./comet-tpch.sh"
```

**Limit to 4 CPUs and 16 GB of memory:**

```shell
docker run --rm \
    --cpus=4 \
    --memory=16g \
    -v /path/to/tpch/data:/data:ro \
    -v /path/to/tpch/queries:/queries:ro \
    -v /path/to/comet-jar-dir:/jars:ro \
    -v /path/to/results:/results \
    -e COMET_JAR=/jars/comet-spark-spark3.5_2.12-0.10.0.jar \
    comet-bench \
    bash -c "./comet-tpch.sh"
```

**Pin to specific CPU cores (useful for avoiding efficiency cores on hybrid CPUs):**

```shell
docker run --rm \
    --cpuset-cpus="0-7" \
    --memory=32g \
    -v /path/to/tpch/data:/data:ro \
    -v /path/to/tpch/queries:/queries:ro \
    -v /path/to/comet-jar-dir:/jars:ro \
    -v /path/to/results:/results \
    -e COMET_JAR=/jars/comet-spark-spark3.5_2.12-0.10.0.jar \
    comet-bench \
    bash -c "./comet-tpch.sh"
```

### Constraint reference

| Docker Flag      | Description                                  | Example        |
| ---------------- | -------------------------------------------- | -------------- |
| `--cpus`         | Number of CPUs (can be fractional)           | `--cpus=8`     |
| `--cpuset-cpus`  | Pin to specific CPU cores                    | `--cpuset-cpus="0-7"` |
| `--memory`       | Maximum memory (hard limit)                  | `--memory=32g` |
| `--memory-swap`  | Total memory + swap (`--memory` to disable swap) | `--memory-swap=32g` |

When setting memory constraints, ensure the limit accommodates both Spark driver memory
(8 GB) and executor memory (16 GB on-heap + 16 GB off-heap) as configured in the
benchmark scripts. A minimum of 32 GB is recommended for the default configurations.

To disable swap (recommended for consistent benchmark results), set `--memory-swap` equal
to `--memory`:

```shell
docker run --rm \
    --cpus=8 \
    --memory=32g \
    --memory-swap=32g \
    ...
```

### Overriding environment variables

You can override any environment variable used by the scripts with `-e`:

```shell
docker run --rm \
    -e SPARK_MASTER=spark://localhost:7077 \
    -e COMET_JAR=/jars/comet.jar \
    -e TPCH_DATA=/data \
    -e TPCH_QUERIES=/queries \
    ...
```

### Comparing Docker-constrained runs

Run benchmarks with different resource limits and compare:

```shell
# Run with 8 CPUs
docker run --rm --cpus=8 --memory=32g \
    -v ... \
    comet-bench bash -c "./comet-tpch.sh && cp *.json /results/"

# Run with 4 CPUs
docker run --rm --cpus=4 --memory=32g \
    -v ... \
    comet-bench bash -c "./comet-tpch.sh && cp *.json /results/"

# Compare results
python3 generate-comparison.py --benchmark tpch \
    --labels "8 CPUs" "4 CPUs" \
    results/comet-tpch-*8cpu*.json results/comet-tpch-*4cpu*.json
```
