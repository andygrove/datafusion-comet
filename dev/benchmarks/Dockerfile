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

# Dockerfile for running Comet TPC-H/TPC-DS benchmarks in a container
# with controllable CPU and memory constraints via `docker run` flags.
#
# See the "Docker Benchmarking" section in README.md for usage instructions.

FROM apache/spark:3.5.8

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        python3 \
        python3-pip \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/opt/spark

# Copy benchmark scripts and Python runner into the image
COPY *.sh *.py /opt/benchmarks/
WORKDIR /opt/benchmarks

# Volume mount points:
#   /data       - TPC-H/TPC-DS data (parquet files)
#   /queries    - SQL query files (q1.sql .. q22.sql / q99.sql)
#   /jars       - Comet JAR (and optionally Iceberg JAR)
#   /results    - Output directory for benchmark JSON results

ENV SPARK_MASTER=spark://localhost:7077
ENV COMET_JAR=/jars/comet.jar
ENV TPCH_DATA=/data
ENV TPCH_QUERIES=/queries
ENV TPCDS_DATA=/data
ENV TPCDS_QUERIES=/queries
