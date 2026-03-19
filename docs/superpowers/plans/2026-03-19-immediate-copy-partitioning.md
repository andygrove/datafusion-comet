# Immediate-Copy Shuffle Partitioning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deferred-gather shuffle partitioning with immediate-copy partitioning using Arrow's `take_record_batch`, eliminating the cache-unfriendly cross-batch gather step.

**Architecture:** During insert, compute partition IDs (same as today via ScratchSpace), then immediately `take_record_batch` per partition and push results to per-partition `Vec<RecordBatch>`. On spill or final write, iterate each partition's Vec and write through BufBatchWriter (coalesce → IPC → compress). Remove PartitionedBatchIterator and interleave_record_batch entirely.

**Tech Stack:** Rust, Arrow (take_record_batch, RecordBatch), DataFusion (MemoryReservation)

**Spec:** `docs/superpowers/specs/2026-03-19-immediate-copy-partitioning-design.md`

---

### Task 1: Change PartitionWriter::spill to accept a slice of RecordBatch

The current `spill()` takes `&mut PartitionedBatchIterator` which we're removing. Change it to accept `&[RecordBatch]`.

**Files:**
- Modify: `native/core/src/execution/shuffle/writers/partition_writer.rs:76-112`
- Modify: `native/core/src/execution/shuffle/partitioners/mod.rs` (remove PartitionedBatchIterator re-export)

- [ ] **Step 1: Change spill signature and body**

In `partition_writer.rs`, change the `spill` method. Replace the `PartitionedBatchIterator` import with `RecordBatch`:

Current (line 19):
```rust
use crate::execution::shuffle::partitioners::PartitionedBatchIterator;
```
Replace with:
```rust
use arrow::array::RecordBatch;
```

Change `spill` method signature and body (lines 76-112):

```rust
    pub(crate) fn spill(
        &mut self,
        batches: &[RecordBatch],
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
        write_buffer_size: usize,
        batch_size: usize,
    ) -> datafusion::common::Result<usize> {
        if batches.is_empty() {
            return Ok(0);
        }

        self.ensure_spill_file_created(runtime)?;

        let total_bytes_written = {
            let mut buf_batch_writer = BufBatchWriter::new(
                &mut self.shuffle_block_writer,
                &mut self.spill_file.as_mut().unwrap().file,
                write_buffer_size,
                batch_size,
            );
            let mut bytes_written = 0;
            for batch in batches {
                bytes_written += buf_batch_writer.write(
                    batch,
                    &metrics.encode_time,
                    &metrics.write_time,
                )?;
            }
            buf_batch_writer.flush(&metrics.encode_time, &metrics.write_time)?;
            bytes_written
        };

        Ok(total_bytes_written)
    }
```

- [ ] **Step 2: Build to verify**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo check 2>&1 | head -20`

This will show errors in `multi_partition.rs` where `spill` is called with the old signature — that's expected and will be fixed in Task 3.

- [ ] **Step 3: Commit**

```bash
git add native/core/src/execution/shuffle/writers/partition_writer.rs
git commit -m "refactor: change PartitionWriter::spill to accept &[RecordBatch]"
```

---

### Task 2: Remove PartitionedBatchIterator

Delete the file and remove its module declaration and re-export.

**Files:**
- Delete: `native/core/src/execution/shuffle/partitioners/partitioned_batch_iterator.rs`
- Modify: `native/core/src/execution/shuffle/partitioners/mod.rs`

- [ ] **Step 1: Delete the file**

```bash
rm native/core/src/execution/shuffle/partitioners/partitioned_batch_iterator.rs
```

- [ ] **Step 2: Update mod.rs**

In `native/core/src/execution/shuffle/partitioners/mod.rs`, remove line 19 and line 26:

```rust
// Remove: mod partitioned_batch_iterator;
// Remove: pub(super) use partitioned_batch_iterator::PartitionedBatchIterator;
```

The file should become:

```rust
mod multi_partition;
mod single_partition;

use arrow::record_batch::RecordBatch;
use datafusion::common::Result;

pub(super) use multi_partition::MultiPartitionShuffleRepartitioner;
pub(super) use single_partition::SinglePartitionShufflePartitioner;

#[async_trait::async_trait]
pub(super) trait ShufflePartitioner: Send + Sync {
    /// Insert a batch into the partitioner
    async fn insert_batch(&mut self, batch: RecordBatch) -> Result<()>;
    /// Write shuffle data and shuffle index file to disk
    fn shuffle_write(&mut self) -> Result<()>;
}
```

- [ ] **Step 3: Commit**

```bash
git add -u native/core/src/execution/shuffle/partitioners/
git commit -m "refactor: remove PartitionedBatchIterator"
```

---

### Task 3: Rewrite MultiPartitionShuffleRepartitioner for immediate-copy

This is the core change. Replace `buffered_batches` + `partition_indices` with `partition_batches: Vec<Vec<RecordBatch>>`, and replace the deferred gather with immediate `take_record_batch`.

**Files:**
- Modify: `native/core/src/execution/shuffle/partitioners/multi_partition.rs`

- [ ] **Step 1: Update imports**

Remove unused imports and add `take_record_batch`:

```rust
// Remove these imports:
// use crate::execution::shuffle::partitioners::partitioned_batch_iterator::{
//     PartitionedBatchIterator, PartitionedBatchesProducer,
// };
// use itertools::Itertools;
// use datafusion::common::utils::proxy::VecAllocExt;

// Change this import (keep ArrayRef, add UInt32Array):
// FROM: use arrow::array::{ArrayRef, RecordBatch};
// TO:
use arrow::array::{ArrayRef, RecordBatch, UInt32Array};

// Add:
use arrow::compute::take_record_batch;
```

Keep all other existing imports. `ArrayRef` is still used in the RoundRobin branch. `VecAllocExt` was only used in `buffer_partitioned_batch_may_spill` which is being deleted.

- [ ] **Step 2: Replace struct fields**

In `MultiPartitionShuffleRepartitioner` struct (lines 109-129), replace:

```rust
    buffered_batches: Vec<RecordBatch>,
    partition_indices: Vec<Vec<(u32, u32)>>,
```

With:

```rust
    /// Per-partition accumulation of take() results. Each Vec holds small
    /// RecordBatches that will be coalesced at write time.
    partition_batches: Vec<Vec<RecordBatch>>,
```

- [ ] **Step 3: Update try_new**

In `try_new` (lines 131-195), replace the initialization of `buffered_batches` and `partition_indices` (lines 182-183):

```rust
            buffered_batches: vec![],
            partition_indices: vec![vec![]; num_output_partitions],
```

With:

```rust
            partition_batches: vec![vec![]; num_output_partitions],
```

- [ ] **Step 4: Replace buffer_partitioned_batch_may_spill**

Replace the entire `buffer_partitioned_batch_may_spill` method (lines 396-436) with a new method that does immediate take + push:

```rust
    fn take_and_store_partitioned_batch(
        &mut self,
        input: &RecordBatch,
        partition_row_indices: &[u32],
        partition_starts: &[u32],
    ) -> datafusion::common::Result<()> {
        let mut mem_growth: usize = 0;
        let num_output_partitions = self.partition_batches.len();

        for partition_id in 0..num_output_partitions {
            let start = partition_starts[partition_id] as usize;
            let end = partition_starts[partition_id + 1] as usize;
            if start == end {
                continue;
            }

            let indices = &partition_row_indices[start..end];
            let indices_array = UInt32Array::from_iter_values(
                indices.iter().copied(),
            );
            let partition_batch = take_record_batch(input, &indices_array)?;

            mem_growth += partition_batch.get_array_memory_size();
            self.partition_batches[partition_id].push(partition_batch);
        }

        if self.reservation.try_grow(mem_growth).is_err() {
            self.spill()?;
        }

        Ok(())
    }
```

- [ ] **Step 5: Update partitioning_batch to call new method**

In `partitioning_batch`, each partitioning variant currently ends with a call to `buffer_partitioned_batch_may_spill`. The call pattern is the same — after computing `partition_starts` and `partition_row_indices`, it calls the buffer method with the input batch.

For the Hash variant (around lines 258-267), change:

```rust
                    // OLD:
                    self.buffer_partitioned_batch_may_spill(
                        input,
                        partition_row_indices,
                        partition_starts,
                    )
                    .await?;
```

To:

```rust
                    self.take_and_store_partitioned_batch(
                        &input,
                        partition_row_indices,
                        partition_starts,
                    )?;
```

Do the same for the RangePartitioning variant and the RoundRobin variant. Search for all calls to `buffer_partitioned_batch_may_spill` and replace with `take_and_store_partitioned_batch`. Note the new method is NOT async (no `.await` needed) and takes `&input` instead of `input` (borrow instead of move).

Since the input batch is no longer moved into `buffered_batches`, it's just borrowed for `take_record_batch`. The batch is dropped after `partitioning_batch` returns.

- [ ] **Step 6: Remove old methods**

Delete the `buffer_partitioned_batch_may_spill` method entirely.

Delete the `partitioned_batches` method (lines 480-489) — it produced `PartitionedBatchesProducer` which no longer exists.

Delete the `shuffle_write_partition` method (lines 438-459) — it took a `PartitionedBatchIterator` which no longer exists. We'll write a replacement in the next step.

- [ ] **Step 7: Rewrite spill()**

Replace the `spill` method (lines 491-525):

```rust
    pub(crate) fn spill(&mut self) -> datafusion::common::Result<()> {
        // Check if there's anything to spill
        let has_data = self.partition_batches.iter().any(|v| !v.is_empty());
        if !has_data {
            return Ok(());
        }

        log::info!(
            "ShuffleRepartitioner spilling shuffle data of {} to disk while inserting ({} time(s) so far)",
            self.used(),
            self.spill_count()
        );

        with_trace("shuffle_spill", self.tracing_enabled, || {
            let num_output_partitions = self.partition_writers.len();
            let mut spilled_bytes = 0;

            for partition_id in 0..num_output_partitions {
                let batches = std::mem::take(
                    &mut self.partition_batches[partition_id],
                );
                spilled_bytes += self.partition_writers[partition_id].spill(
                    &batches,
                    &self.runtime,
                    &self.metrics,
                    self.write_buffer_size,
                    self.batch_size,
                )?;
            }

            self.reservation.free();
            self.metrics.spill_count.add(1);
            self.metrics.spilled_bytes.add(spilled_bytes);
            Ok(())
        })
    }
```

- [ ] **Step 8: Rewrite shuffle_write()**

Replace the `shuffle_write` method (lines 559-630). The new version writes in-memory batches through a temporary `BufBatchWriter` instead of using `PartitionedBatchIterator`:

```rust
    fn shuffle_write(&mut self) -> datafusion::common::Result<()> {
        with_trace("shuffle_write", self.tracing_enabled, || {
            let start_time = Instant::now();

            let num_output_partitions = self.partition_batches.len();
            let mut offsets = vec![0; num_output_partitions + 1];

            let data_file = self.output_data_file.clone();
            let index_file = self.output_index_file.clone();

            let output_data = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(data_file)
                .map_err(|e| {
                    DataFusionError::Execution(format!("shuffle write error: {e:?}"))
                })?;

            let mut output_data = BufWriter::new(output_data);

            for i in 0..num_output_partitions {
                offsets[i] = output_data.stream_position()?;

                // Copy spill file contents if any
                if let Some(spill_path) = self.partition_writers[i].path() {
                    let mut spill_file = BufReader::new(File::open(spill_path)?);
                    let mut write_timer = self.metrics.write_time.timer();
                    std::io::copy(&mut spill_file, &mut output_data)?;
                    write_timer.stop();
                }

                // Write remaining in-memory batches
                let batches = std::mem::take(&mut self.partition_batches[i]);
                if !batches.is_empty() {
                    let mut buf_batch_writer = BufBatchWriter::new(
                        &mut self.shuffle_block_writer,
                        &mut output_data,
                        self.write_buffer_size,
                        self.batch_size,
                    );
                    for batch in &batches {
                        buf_batch_writer.write(
                            batch,
                            &self.metrics.encode_time,
                            &self.metrics.write_time,
                        )?;
                    }
                    buf_batch_writer.flush(
                        &self.metrics.encode_time,
                        &self.metrics.write_time,
                    )?;
                }
            }

            let mut write_timer = self.metrics.write_time.timer();
            output_data.flush()?;
            write_timer.stop();

            offsets[num_output_partitions] = output_data.stream_position()?;

            let mut write_timer = self.metrics.write_time.timer();
            let mut output_index =
                BufWriter::new(File::create(index_file).map_err(|e| {
                    DataFusionError::Execution(format!("shuffle write error: {e:?}"))
                })?);
            for offset in offsets {
                output_index.write_all(&(offset as i64).to_le_bytes()[..])?;
            }
            output_index.flush()?;
            write_timer.stop();

            self.metrics
                .baseline
                .elapsed_compute()
                .add_duration(start_time.elapsed());

            Ok(())
        })
    }
```

- [ ] **Step 9: Update Debug impl**

The `Debug` impl (lines 633-642) references `self.used()` etc. which still work. No change needed unless `partition_indices` was referenced — it wasn't.

- [ ] **Step 10: Build**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo check`
Expected: Compiles successfully. Fix any remaining import or type errors.

- [ ] **Step 11: Commit**

```bash
git add native/core/src/execution/shuffle/partitioners/multi_partition.rs
git commit -m "feat: replace deferred-gather with immediate-copy partitioning"
```

---

### Task 4: Run Rust tests

**Files:** (no changes — verification only)

- [ ] **Step 1: Run Rust tests**

```bash
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo test
```

Expected: All tests pass. The key tests are in `shuffle_writer.rs`:
- `shuffle_partitioner_memory` (tests insert + spill with memory limits)
- `test_round_robin_deterministic` (tests round robin produces identical results)

- [ ] **Step 2: Run clippy**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo clippy --all-targets --workspace -- -D warnings
```

Expected: No warnings.

- [ ] **Step 3: Fix any failures and commit**

```bash
git add -u && git commit -m "fix: address test/clippy issues"
```

---

### Task 5: Run JVM shuffle tests

**Files:** (no changes — verification only)

- [ ] **Step 1: Build native code**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && make core
```

- [ ] **Step 2: Run native shuffle test suite**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -Dsuites="org.apache.comet.exec.CometNativeShuffleSuite"
```

Expected: All tests pass. These exercise hash, range, round-robin, and single partitioning with various data types.

- [ ] **Step 3: Run broader exec suite**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -Dsuites="org.apache.comet.exec.CometExecSuite"
```

Expected: All tests pass.

- [ ] **Step 4: Fix any failures**

If a test fails, check whether it's related to partitioning correctness (wrong rows in wrong partitions) or spill behavior. Add logging to `take_and_store_partitioned_batch` if needed to debug.

---

### Task 6: Run format checks and clean up

- [ ] **Step 1: Format**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && make format
```

- [ ] **Step 2: Commit any formatting changes**

```bash
git add -u && git commit -m "style: format code"
```

- [ ] **Step 3: Remove itertools dependency if no longer needed**

Check if `itertools` is still used elsewhere in the crate:

```bash
cd /Users/andy/git/apache/temp/datafusion-comet/native && grep -r "use itertools" core/src/
```

If `multi_partition.rs` was the only user (it used `Itertools` for `tuple_windows`), remove `itertools` from `native/core/Cargo.toml`. If still used elsewhere, leave it.

- [ ] **Step 4: Commit if changed**

```bash
git add -u && git commit -m "chore: remove unused itertools dependency"
```
