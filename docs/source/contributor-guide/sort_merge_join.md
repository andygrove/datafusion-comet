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

### v2 (Built-in Batch Coalescing)

DataFusion's SMJ with join filters produces tiny batches because filtering removes rows from already-small output batches. Previously, the planner wrapped filtered SMJ with `CoalesceBatchesExec` to coalesce these. Now coalescing is built directly into `CometSortMergeJoinStream`:

1. **`Exhausted` state**: Instead of emitting filtered batches directly, merges them into `self.output` (the accumulator that the `Init` state already uses) before emitting. This ensures the final batch is coalesced with any previously accumulated rows.
2. **Removed `CoalesceBatchesExec` wrapper**: The planner no longer wraps filtered SMJ with `CoalesceBatchesExec`, eliminating one operator and one `concat_batches` copy from the query plan.

The `Init` state already had coalescing logic (accumulating filtered batches into `self.output` and only emitting when `num_rows >= batch_size`). The v2 change ensures the `Exhausted` state uses the same accumulator instead of bypassing it.

### Integration Points

**Planner** (`native/core/src/execution/planner.rs`): The `OpStruct::SortMergeJoin` match arm creates `CometSortMergeJoinExec` instead of `SortMergeJoinExec`. The `NullEquality::NullEqualsNothing` argument is no longer passed (hardcoded inside). No `CoalesceBatchesExec` wrapping — coalescing is built into the join stream.

**Module registration** (`native/core/src/execution/operators/mod.rs`): Declares `mod sort_merge_join` and `pub use sort_merge_join::CometSortMergeJoinExec`.

**No protobuf changes**: The `SortMergeJoin` protobuf message is unchanged.

**No Scala changes**: The Scala `CometSortMergeJoinExec` wrapper class is unchanged.

## Supported Join Types

All DataFusion join types are supported:

- Inner, Left, Right, Full (outer)
- LeftSemi, RightSemi, LeftAnti, RightAnti
- LeftMark, RightMark

## Memory Management and Spilling

### Memory Reservation

Each partition's `CometSortMergeJoinStream` registers a `MemoryConsumer` named `CometSMJStream[{partition}]` with DataFusion's memory pool. The reservation tracks memory used by **buffered-side batches only** — the streamed side processes one batch at a time and does not reserve memory.

When a new buffered batch arrives, `allocate_reservation()` calls `reservation.try_grow(size_estimation)`. The size estimation includes:

- The `RecordBatch` array memory (`get_array_memory_size()`)
- Join key arrays extracted for the batch
- Per-row overhead for bookkeeping (`num_rows.next_power_of_two() * size_of::<usize>()`)

The `peak_mem_used` gauge metric records the high-water mark of the reservation.

When a buffered batch is no longer needed (its join key group is fully processed), `free_reservation()` shrinks the reservation by the batch's `size_estimation`.

### Spill Trigger and Mechanism

Spilling is triggered when `try_grow()` fails (the memory pool cannot accommodate the new buffered batch):

1. **Check disk manager**: If `runtime_env.disk_manager.tmp_files_enabled()` is true, spilling proceeds. Otherwise, the error propagates as `"<error>. Disk spilling disabled."`.

2. **Spill to IPC file**: The in-memory `RecordBatch` is written to a temporary file via `SpillManager::spill_record_batch_and_finish()`. Files are named `comet_sort_merge_join_buffered_spill`. The `SpillManager` uses Arrow IPC format with optional compression (configured via `session_config().spill_compression()`).

3. **State transition**: The `BufferedBatch` state changes from `BufferedBatchState::InMemory(RecordBatch)` to `BufferedBatchState::Spilled(RefCountedTempFile)`. The batch's `join_arrays` (join key columns) remain in memory for comparison — only the full row data is spilled.

4. **Read-back on demand**: When a spilled batch is needed for output (in `fetch_right_columns_from_batch_by_idxs()`), the IPC file is re-read via Arrow's `StreamReader` and the required columns are extracted with `take()`. This read happens each time the batch is accessed — there is no caching of re-read data.

### What Spills and What Doesn't

| Data | Spills? | Why |
| ---- | ------- | --- |
| Buffered-side batches (row data) | Yes | Multiple batches with the same join key accumulate; can grow unboundedly for skewed keys |
| Buffered-side join key arrays | No | Kept in memory for key comparison during scanning |
| Streamed-side batch | No | Only one batch at a time; replaced as the stream advances |
| `staging_output_record_batches` | No | Intermediate join output before filtering; bounded by `batch_size` |
| `self.output` accumulator | No | Coalesced filtered output; bounded by `batch_size` |

### Metrics

| Metric | Type | Description |
| ------ | ---- | ----------- |
| `join_time` | Time | Total wall time in the join state machine |
| `input_batches` | Count | Batches consumed from both input streams |
| `input_rows` | Count | Rows consumed from both input streams |
| `output_batches` | Count | Batches emitted by the join |
| `peak_mem_used` | Gauge | High-water mark of the memory reservation (bytes) |
| `spill_count` | Count | Number of batches spilled to disk (from `SpillMetrics`) |
| `spill_size` | Count | Total bytes written to spill files (from `SpillMetrics`) |

## Verification

```bash
# Build native code
make core

# Full integration tests (from Comet root)
./mvnw test -Dsuites="org.apache.comet.exec.CometJoinSuite" -Dtest=none
```

## Potential Performance Improvements

### Null-Key Batch Pre-Filtering

**Impact**: Medium (depends on null density in join keys)

In Spark workloads with nullable foreign keys, many streamed rows may have NULL join keys. Since Comet hardcodes `NullEquality::NullEqualsNothing`, these rows can never match any buffered row. Currently, each null-key streamed row still goes through `compare_streamed_buffered()` and the full state machine cycle.

**Optimization**: Before entering the `Polling` state, check if the current streamed row's join key columns are all null. If so, skip directly to producing the unmatched output (for Left/Full/LeftAnti joins) or skip entirely (for Inner/LeftSemi joins). For batches where a large fraction of join keys are null, this avoids unnecessary buffered-side comparisons and key array evaluations.

**Where to change**: `CometSortMergeJoinStream::poll_next()` in the `Init` → `Polling` transition, and potentially `compare_streamed_buffered()`.

### Buffer Reuse for Consecutive Duplicate Keys

**Impact**: High (for skewed joins with many duplicate keys)

When consecutive streamed rows share the same join key, the current implementation re-scans the buffered data for each row: `scanning_reset()` is called in the `Init` state, and `join_partial()` iterates through all buffered batches again. For joins with high key cardinality on the streamed side (e.g., fact-dimension joins where many fact rows share the same dimension key), this causes redundant work.

**Optimization**: Detect when the current streamed row has the same join key as the previous one. In this case, reuse the already-matched buffered output indices from the previous iteration instead of re-scanning. This mirrors Spark's sort-merge join optimization for skewed keys.

**Where to change**: `CometSortMergeJoinStream` needs a `previous_streamed_key` field. In the `Init` state, compare the new streamed key against the previous one; if equal, skip to `JoinOutput` with the same buffered data instead of resetting.

### ExistenceJoin Support

**Impact**: Medium (enables native execution of correlated subqueries)

Spark has an `ExistenceJoin` type used in correlated subqueries (`WHERE EXISTS (SELECT ...)`). DataFusion doesn't support this join type natively, so Comet currently falls back to Spark for these queries. Adding `ExistenceJoin` to `CometSortMergeJoinExec` would allow native execution.

**Behavior**: `ExistenceJoin` is similar to `LeftMark` — it appends a boolean column indicating whether a match was found — but with Spark-specific semantics for null handling in the existence column.

**Where to change**: Add `ExistenceJoin` to the `join_type` match arms throughout the stream (similar to `LeftMark`), and update the Scala serde to map Spark's `ExistenceJoin` to the native operator.

### Spilled Batch Caching

**Impact**: Medium-High (for memory-constrained workloads with large buffered sides)

When buffered batches are spilled to disk and then needed for output, `fetch_right_columns_from_batch_by_idxs()` re-reads the entire IPC file from disk via Arrow's `StreamReader` on every access. For join keys with many matching rows, the same spilled batch may be read multiple times — once per streamed row that matches the key group.

**Optimization**: Cache the most recently read-back spilled batch in memory (a single-entry LRU). Since all buffered batches in `BufferedData` share the same join key, they are typically accessed sequentially and then freed. A single cached batch would eliminate most redundant disk reads without significantly increasing memory usage.

**Where to change**: `fetch_right_columns_from_batch_by_idxs()` and `BufferedBatch` (add an optional cached `RecordBatch` field).

### Vectorized Key Comparison

**Impact**: Medium (reduces per-row overhead in key comparison)

Currently, `compare_join_arrays()` compares one streamed row against one buffered row at a time, and `is_join_arrays_equal()` similarly checks one pair. For multi-column join keys or wide key types (strings, decimals), the per-row function call overhead is significant.

**Optimization**: Batch the comparison — instead of comparing one streamed row at a time, compare a range of streamed rows against the current buffered key using vectorized Arrow compute kernels (`eq`, `cmp`). This amortizes function call overhead and enables SIMD for fixed-width key types.

**Where to change**: `compare_streamed_buffered()` and the `Polling` → `JoinOutput` transition. Requires rethinking the single-row state machine to support multi-row advancement.

### Reduce `concat_batches` Copies in Filtered Path

**Impact**: Low-Medium (reduces memory copies for filtered joins)

The filtered join path involves multiple `concat_batches` calls:
1. `output_record_batch_and_reset()` concats staging batches (called for bookkeeping even though the result is dropped for filtered joins)
2. `filter_joined_batch()` concats staging batches again
3. The `Init` state concats the filtered result into `self.output`

**Optimization**: For filtered joins, skip the `output_record_batch_and_reset()` call in the `JoinOutput` state and track `output_size` differently (e.g., sum of staging batch row counts). This eliminates one redundant `concat_batches` on the hot path. The `filter_joined_batch()` concat is unavoidable since filtering needs a contiguous batch, but the `Init` state accumulation into `self.output` could use a `Vec<RecordBatch>` instead of a single `RecordBatch` to defer concatenation until emission.

**Where to change**: `JoinOutput` state (lines 1041-1058), `output_record_batch_and_reset()`, and `self.output` type.

### Spark-Compatible Metrics

**Impact**: Low (observability improvement, no performance change)

Add metrics that align with Spark's sort-merge join reporting for easier debugging and performance analysis:

- **Spill metrics**: Spill count and size are already tracked via `SpillMetrics` but could be surfaced more prominently in explain plans.
- **Buffer utilization**: Track the number of buffered batches per join key group (helps identify skew).
- **Join key cardinality**: Count distinct join key values processed (helps estimate join selectivity).
- **Filter selectivity**: For filtered joins, track the ratio of rows that pass the filter vs total joined rows.

**Where to change**: `CometSortMergeJoinMetrics` and the various state machine transitions.

### ~~Built-in Batch Coalescing~~ (Done in v2)

Built-in coalescing is now part of the join stream. The `Exhausted` state merges filtered output into the accumulator before emitting, and the planner no longer wraps with `CoalesceBatchesExec`.

## Changelog

| Date       | Change             | Details                                                              |
| ---------- | ------------------ | -------------------------------------------------------------------- |
| 2026-02-24 | Initial fork (v1)  | Fork DataFusion 51.0.0 SMJ, rename, hardcode null equality, simplify |
| 2026-02-24 | Batch coalescing (v2) | Built-in coalescing in Exhausted state, removed CoalesceBatchesExec wrapper |
