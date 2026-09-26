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

# Comet Upgrade Guide

This guide lists the behavior changes in each Comet release and the configuration setting that
restores the previous behavior for each one. Read the section for every version between the release
you are upgrading from and the release you are upgrading to.

A **behavior change** is one where the same query, run over the same data, with the same explicitly
set configuration, produces a different result or a different error than it did in the previous
release. Comet's [versioning policy](../../about/versioning_policy.md) permits these in a minor
release only when a `spark.comet.legacy.*` configuration key restores the previous behavior, so
every entry below names such a key.

Two kinds of change are deliberately absent from this guide:

- **Correctness fixes.** Comet's goal is to return the results Apache Spark returns. When Comet
  returns something different for an expression or operator marked `Compatible`, that is a bug, and
  fixing it is a bug fix rather than a behavior change. These fixes appear in the release notes, not
  here. For a fix with an unusually wide blast radius the maintainers may still provide a
  `spark.comet.legacy.*` key, in which case it will be listed below.
- **Changes to which operators run natively.** Whether a given expression runs in Comet or falls
  back to Spark can change in any release. This affects performance, not results.

Additions, new configuration keys, and Apache Spark version support changes are recorded in the
release notes and on the
[Spark Version Compatibility](compatibility/spark-versions.md) page rather than here.

## Legacy Configuration Keys

Every key under `spark.comet.legacy.*` is deprecated from the moment it is added. Each one exists to
give you time to adapt to a behavior change, and may be removed in any future major release, at
which point the newer behavior becomes unconditional.

Treat setting one of these keys as a temporary measure. If you find you cannot stop relying on a
legacy behavior, please open an issue describing your use case so it can be considered before the
key is removed.

## Upgrading to Comet 1.2.0

### JVM-side Arrow memory is charged to Spark's off-heap memory pool

The Arrow memory that Comet allocates on the JVM side for a task, for example to read a cached table
or to exchange batches with a Python worker, is now charged to `spark.memory.offHeap.size` as a
memory consumer of that task. An allocation the pool cannot cover fails the task with an
`OutOfMemoryException` whose message begins `Unable to reserve`, the way a Spark operator fails when
it cannot acquire memory. The charge also leaves Comet's native operators and Spark's own consumers
less of the pool, so queries can spill sooner than before. Earlier releases charged this memory to no
pool, so it never failed an allocation and had to fit in `spark.executor.memoryOverhead` instead.

Arrow memory that is handed to Comet's native code, or received from it, is not charged on the JVM
side, because native accounts for what it retains.

To restore the previous behavior, set `spark.comet.legacy.unboundedJvmArrowMemory=true` when the
application is submitted. It is read on the executors, so setting it in a running session has no
effect.

## Upgrading to Comet 1.1.0

Comet `1.1.0` makes no behavior changes that need a `spark.comet.legacy.*` key.

Comet `1.1.0` requires JDK 17 or later. JDK 11 is no longer supported. See
[Installing Comet](installation.md) for the supported Java, Scala, and Spark versions.

## Upgrading to Comet 1.0.0

Comet `1.0.0` is the first release under the stable
[versioning policy](../../about/versioning_policy.md). From this release onward, behavior changes
are documented on this page along with the configuration key that reverts each one.

Changes made during the `0.x` series are not recorded here. If you are upgrading from a `0.x`
release, review the release notes for the versions in between.
