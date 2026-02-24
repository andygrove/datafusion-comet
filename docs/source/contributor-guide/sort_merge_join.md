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

# CometSortMergeJoinExec

## Overview

`CometSortMergeJoinExec` is a custom Sort-Merge Join operator for DataFusion Comet, forked from DataFusion 51.0.0's `SortMergeJoinExec`. It provides a Spark-optimized join implementation that we can iteratively improve with Spark-specific semantics.

**Source**: `native/core/src/execution/operators/sort_merge_join.rs`
**Forked from**: DataFusion 51.0.0 (`datafusion-physical-plan` crate, `joins/sort_merge_join/` module)

## Motivation

Comet previously used DataFusion's `SortMergeJoinExec` directly. By forking, we gain:

1. **Null-key hardcoding**: Spark always uses `NullEquality::NullEqualsNothing` — no need to parameterize
2. **Explain plan clarity**: Shows as `CometSortMergeJoinExec` in query plans
3. **Optimization surface**: Ability to add Spark-specific fast paths without upstream constraints
4. **Metrics control**: Future Spark-compatible spill/buffer metrics

## Architecture

### File Structure

The fork consolidates three DataFusion source files into one:

| DataFusion Source                           | Content                                                          |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `joins/sort_merge_join/exec.rs` (~594 lines)   | `SortMergeJoinExec` plan node, `DisplayAs`, `ExecutionPlan` impl |
| `joins/sort_merge_join/stream.rs` (~1,941 lines) | `SortMergeJoinStream`, state machines, join logic                |
| `joins/sort_merge_join/metrics.rs` (~97 lines)  | `SortMergeJoinMetrics`                                           |

All merged into: `sort_merge_join.rs` with `Comet` prefixes.

### Key Types

| Type                        | Role                                                              |
| --------------------------- | ----------------------------------------------------------------- |
| `CometSortMergeJoinExec`    | `ExecutionPlan` node — the plan-level operator                    |
| `CometSortMergeJoinStream`  | `Stream<Item=Result<RecordBatch>>` — runtime state machine        |
| `CometSortMergeJoinMetrics` | Metrics: join_time, input/output batches, peak memory, spill      |
| `StreamedBatch`             | Current streamed-side batch with join key arrays and output indices |
| `BufferedBatch`             | Buffered-side batch (in-memory or spilled to disk)                |
| `BufferedData`              | Collection of buffered batches sharing one join key               |

### State Machine

The stream operates as a state machine with four states:

```
Init → Polling → JoinOutput → Exhausted
  ↑        ↓          ↓
  └────────┴──────────┘
```

- **Init**: Decide whether to advance streamed or buffered side
- **Polling**: Poll next streamed row and/or buffered batches
- **JoinOutput**: Match streamed rows with buffered rows, produce output
- **Exhausted**: Flush remaining output

### Streamed vs Buffered

The probe (streamed) side is determined by join type:

- **Left probe**: Inner, Left, Full, LeftAnti, LeftSemi, LeftMark
- **Right probe**: Right, RightSemi, RightAnti, RightMark

The buffered side accumulates all rows with the same join key. If memory limits are exceeded and spilling is enabled, buffered batches spill to disk.

## Changes from DataFusion

### v1 (Initial Fork)

1. **Renamed** all types: `SortMergeJoin*` → `CometSortMergeJoin*`
2. **Removed `null_equality` parameter** from `try_new()` — hardcoded to `NullEquality::NullEqualsNothing` in `compare_streamed_buffered()`
3. **Removed `swap_inputs()`** — not needed for Comet's planner
4. **Removed `try_swapping_with_projection()`** — projection pushdown optimization not needed
5. **Simplified `statistics()`** — returns unknown (DataFusion's `estimate_join_statistics` is `pub(crate)`)
6. **Simplified `compute_properties()`** — inline boundedness check and output partitioning (DataFusion's `symmetric_join_output_partitioning` and `boundedness_from_children` are `pub(crate)`)
7. **Display**: Shows `CometSortMergeJoin` instead of `SortMergeJoin`, removed NullEquality display (always NullEqualsNothing)
8. **Import paths**: Uses `datafusion::` umbrella crate paths instead of direct `datafusion_common::`, `datafusion_execution::`, etc.

### Integration Points

**Planner** (`native/core/src/execution/planner.rs`): The `OpStruct::SortMergeJoin` match arm creates `CometSortMergeJoinExec` instead of `SortMergeJoinExec`. The `NullEquality::NullEqualsNothing` argument is no longer passed (hardcoded inside).

**Module registration** (`native/core/src/execution/operators/mod.rs`): Declares `mod sort_merge_join` and `pub use sort_merge_join::CometSortMergeJoinExec`.

**No protobuf changes**: The `SortMergeJoin` protobuf message is unchanged.

**No Scala changes**: The Scala `CometSortMergeJoinExec` wrapper class is unchanged.

## Supported Join Types

All DataFusion join types are supported:

- Inner, Left, Right, Full (outer)
- LeftSemi, RightSemi, LeftAnti, RightAnti
- LeftMark, RightMark

## Spilling

Buffered batches can spill to disk when memory limits are exceeded:

- Uses DataFusion's `SpillManager` for IPC-based spill/read
- Spill files named `comet_sort_merge_join_buffered_spill`
- Only buffered side spills; streamed side stays in memory

## Verification

```bash
# Build native code
make core

# Full integration tests (from Comet root)
./mvnw test -Dsuites="org.apache.comet.exec.CometJoinSuite" -Dtest=none
```

## Future Optimization Path

### Phase 2: Null-Key Pre-Filtering

Skip entire streamed batches where all join key columns are null. In Spark workloads with nullable foreign keys, this avoids unnecessary buffered-side comparisons.

### Phase 3: Buffer Reuse for Consecutive Duplicate Keys

When consecutive streamed rows have identical join keys, reuse the already-matched buffered output instead of re-scanning. Mirrors Spark's optimization for skewed joins.

### Phase 4: ExistenceJoin Support

Spark has an `ExistenceJoin` type (used in correlated subqueries) that DataFusion doesn't support. Adding native support would allow Comet to handle these without fallback to Spark.

### Phase 5: Custom Metrics

Add Spark-compatible metrics:

- Spill size / spill count aligned with Spark's reporting
- Buffer utilization metrics
- Join key cardinality tracking

### Phase 6: Built-in Batch Coalescing

Currently, filtered SMJ output is wrapped with `CoalesceBatchesExec`. Building coalescing directly into the join stream eliminates the overhead of an extra operator and reduces memory copies.

## Changelog

| Date       | Change       | Details                                                              |
| ---------- | ------------ | -------------------------------------------------------------------- |
| 2026-02-24 | Initial fork | Fork DataFusion 51.0.0 SMJ, rename, hardcode null equality, simplify |
