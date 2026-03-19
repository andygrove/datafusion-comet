# Custom Scatter Kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deferred-gather shuffle partitioning (interleave_record_batch, 55% of shuffle write time) with a custom scatter kernel that appends values directly to per-partition typed buffers in a single pass during insert.

**Architecture:** New `PartitionBuffer` struct with typed `ColumnBuffer` variants (Fixed, Variable, LargeVariable, Boolean, Fallback). During insert, scatter values from each input column to partition buffers using partition_ids from ScratchSpace. Buffers auto-flush to RecordBatch (zero-copy) when reaching batch_size, then write through existing IPC+compress pipeline. Remove PartitionedBatchIterator and interleave_record_batch entirely.

**Tech Stack:** Rust, Arrow (MutableBuffer, BooleanBufferBuilder, ScalarBuffer, NullBuffer, array constructors), DataFusion (MemoryReservation)

**Spec:** `docs/superpowers/specs/2026-03-19-scatter-kernel-design.md`

---

### Task 1: Create PartitionBuffer and ColumnBuffer

New file containing the per-partition typed buffer structure. This is the core data structure — no integration yet.

**Files:**
- Create: `native/core/src/execution/shuffle/partitioners/partition_buffer.rs`
- Modify: `native/core/src/execution/shuffle/partitioners/mod.rs` (add module)

- [ ] **Step 1: Create the file with ColumnBuffer enum and PartitionBuffer struct**

The file should contain:
- `ColumnBuffer` enum with variants: `Boolean`, `Fixed`, `Variable`, `LargeVariable`, `Fallback`
- `PartitionBuffer` struct with `columns: Vec<ColumnBuffer>`, `row_count: usize`, `schema: SchemaRef`
- `PartitionBuffer::new(schema, estimated_rows)` — creates appropriate ColumnBuffer variant per column
- `PartitionBuffer::memory_size()` — sum of all buffer capacities
- `PartitionBuffer::row_count()` accessor
- `PartitionBuffer::clear()` — reset all buffers
- `PartitionBuffer::has_fallback_columns()` — check if any column uses Fallback

For `ColumnBuffer`, implement append methods:
- `Fixed`: `append_fixed(&mut self, bytes: &[u8])` — extend MutableBuffer
- `Variable`: `append_variable(&mut self, bytes: &[u8])` — push offset, extend data
- `LargeVariable`: `append_large_variable(&mut self, bytes: &[u8])` — same with i64
- `Boolean`: `append_bool(&mut self, value: bool)` — append to BooleanBufferBuilder
- `Fallback`: `append_fallback_index(&mut self, idx: u32)` — push index
- All variants: `append_null_bit(&mut self, is_valid: bool)` — append to nulls builder

For type mapping in `new()`, use the spec's table:
```rust
match field.data_type() {
    DataType::Boolean => ColumnBuffer::Boolean { ... },
    DataType::Int8 | DataType::UInt8 => ColumnBuffer::Fixed { byte_width: 1, ... },
    DataType::Int16 | DataType::UInt16 | DataType::Float16 => ColumnBuffer::Fixed { byte_width: 2, ... },
    DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => ColumnBuffer::Fixed { byte_width: 4, ... },
    DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Date64
    | DataType::Timestamp(_, _) | DataType::Duration(_) => ColumnBuffer::Fixed { byte_width: 8, ... },
    DataType::Decimal128(_, _) => ColumnBuffer::Fixed { byte_width: 16, ... },
    DataType::Utf8 | DataType::Binary => ColumnBuffer::Variable {
        offsets: vec![0i32], data: vec![], nulls: BooleanBufferBuilder::new(estimated_rows),
    },
    DataType::LargeUtf8 | DataType::LargeBinary => ColumnBuffer::LargeVariable {
        offsets: vec![0i64], data: vec![], nulls: BooleanBufferBuilder::new(estimated_rows),
    },
    _ => ColumnBuffer::Fallback { ... },
}
```

For `memory_size()`:
```rust
match self {
    Fixed { values, nulls, .. } => values.capacity() + nulls.capacity(),
    Variable { offsets, data, nulls } => offsets.capacity() * 4 + data.capacity() + nulls.capacity(),
    // ... similar for others
}
```

- [ ] **Step 2: Add module declaration in mod.rs**

Add `mod partition_buffer;` and a `pub(super) use` for `PartitionBuffer`.

- [ ] **Step 3: Build**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo check`

- [ ] **Step 4: Commit**

```bash
git add native/core/src/execution/shuffle/partitioners/partition_buffer.rs \
        native/core/src/execution/shuffle/partitioners/mod.rs
git commit -m "feat: add PartitionBuffer with typed ColumnBuffer for scatter kernel"
```

---

### Task 2: Add flush() to PartitionBuffer

Implement `flush()` which constructs a RecordBatch from the accumulated buffers. This is where the zero-copy conversion happens.

**Files:**
- Modify: `native/core/src/execution/shuffle/partitioners/partition_buffer.rs`

- [ ] **Step 1: Implement flush**

`flush(&mut self, fallback_batch: Option<&RecordBatch>) -> Result<RecordBatch>`

For each column, construct an ArrayRef:

```rust
fn flush_column(col: &mut ColumnBuffer, row_count: usize, data_type: &DataType,
                fallback_batch: Option<&RecordBatch>, col_idx: usize) -> Result<ArrayRef> {
    match col {
        ColumnBuffer::Fixed { values, byte_width, nulls } => {
            let null_buffer = if nulls.len() > 0 {
                Some(NullBuffer::new(nulls.finish()))
            } else {
                None
            };
            let buffer = ScalarBuffer::from(std::mem::replace(values, MutableBuffer::new(0)));
            // Use make_array or construct typed array based on data_type
            // e.g., for Int32: Arc::new(Int32Array::new(buffer.into(), null_buffer))
            // Generic approach: use ArrayData::builder
            let array_data = ArrayData::builder(data_type.clone())
                .len(row_count)
                .add_buffer(buffer.into_inner())
                .null_bit_buffer(null_buffer.map(|n| n.into_inner().into_inner()))
                .build()?;
            Ok(make_array(array_data))
        }
        ColumnBuffer::Variable { offsets, data, nulls } => {
            // Add final offset
            offsets.push(data.len() as i32);
            let offset_buffer = OffsetBuffer::new(ScalarBuffer::from(
                std::mem::take(offsets)));
            let values_buffer = Buffer::from(std::mem::take(data));
            let null_buffer = if nulls.len() > 0 {
                Some(NullBuffer::new(nulls.finish()))
            } else {
                None
            };
            match data_type {
                DataType::Utf8 => Ok(Arc::new(StringArray::new(
                    offset_buffer, values_buffer, null_buffer))),
                DataType::Binary => Ok(Arc::new(BinaryArray::new(
                    offset_buffer, values_buffer, null_buffer))),
                _ => unreachable!(),
            }
        }
        // Similar for LargeVariable, Boolean
        ColumnBuffer::Fallback { indices } => {
            let indices_array = UInt32Array::from(std::mem::take(indices));
            let batch = fallback_batch.expect("Fallback flush requires input batch");
            let taken = take(batch.column(col_idx), &indices_array, None)?;
            Ok(taken)
        }
    }
}
```

After flushing, reset row_count to 0. Reinitialize offset vectors with initial `[0]` for Variable types.

- [ ] **Step 2: Write a unit test**

Add `#[cfg(test)]` module with a test that:
1. Creates a PartitionBuffer for a schema with Int32, Utf8, Boolean columns
2. Appends several rows using the append methods
3. Calls `flush(None)`
4. Asserts the resulting RecordBatch has correct values, nulls, and row count

- [ ] **Step 3: Run test**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && DYLD_LIBRARY_PATH=$JAVA_HOME/lib/server cargo test partition_buffer`

- [ ] **Step 4: Commit**

```bash
git add native/core/src/execution/shuffle/partitioners/partition_buffer.rs
git commit -m "feat: add flush() to PartitionBuffer with zero-copy RecordBatch construction"
```

---

### Task 3: Change PartitionWriter::spill signature

Same change as in the previous plan — change from `&mut PartitionedBatchIterator` to `&[RecordBatch]`.

**Files:**
- Modify: `native/core/src/execution/shuffle/writers/partition_writer.rs`

- [ ] **Step 1: Change spill signature and implementation**

Replace the `PartitionedBatchIterator` import with `RecordBatch`:
```rust
use arrow::array::RecordBatch;
```

Change `spill` to accept `&[RecordBatch]`:
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
                batch, &metrics.coalesce_time, &metrics.encode_time, &metrics.write_time)?;
        }
        buf_batch_writer.flush(&metrics.coalesce_time, &metrics.encode_time, &metrics.write_time)?;
        bytes_written
    };
    Ok(total_bytes_written)
}
```

Note: This references `coalesce_time` from the profiling metrics PR. If that hasn't been merged to the branch you're on, use `encode_time` for the coalesce parameter too (or add a dummy Time). Check what's in the current `BufBatchWriter::write` signature.

- [ ] **Step 2: Commit** (will have compile errors until Task 4)

```bash
git add native/core/src/execution/shuffle/writers/partition_writer.rs
git commit -m "refactor: change PartitionWriter::spill to accept &[RecordBatch]"
```

---

### Task 4: Rewrite MultiPartitionShuffleRepartitioner with scatter kernel

This is the core integration. Replace buffered_batches + partition_indices with partition_buffers, replace buffer_partitioned_batch_may_spill with scatter_batch.

**Files:**
- Modify: `native/core/src/execution/shuffle/partitioners/multi_partition.rs`
- Delete: `native/core/src/execution/shuffle/partitioners/partitioned_batch_iterator.rs`
- Modify: `native/core/src/execution/shuffle/partitioners/mod.rs`

- [ ] **Step 1: Update imports in multi_partition.rs**

Remove:
```rust
use crate::execution::shuffle::partitioners::partitioned_batch_iterator::{
    PartitionedBatchIterator, PartitionedBatchesProducer,
};
use datafusion::common::utils::proxy::VecAllocExt;
use itertools::Itertools;
use datafusion::physical_plan::metrics::Time;
```

Add:
```rust
use crate::execution::shuffle::partitioners::partition_buffer::PartitionBuffer;
```

Keep `ArrayRef` (used in RoundRobin).

- [ ] **Step 2: Replace struct fields**

Replace:
```rust
    buffered_batches: Vec<RecordBatch>,
    partition_indices: Vec<Vec<(u32, u32)>>,
```

With:
```rust
    partition_buffers: Vec<PartitionBuffer>,
    has_fallback_columns: bool,
```

- [ ] **Step 3: Update try_new**

Replace `buffered_batches` and `partition_indices` initialization with:
```rust
    let has_fallback_columns = schema.fields().iter().any(|f| {
        matches!(f.data_type(),
            DataType::Struct(_) | DataType::List(_) | DataType::LargeList(_)
            | DataType::Map(_, _) | DataType::Dictionary(_, _) | DataType::Decimal256(_, _)
            // ... any other Fallback types
        )
    });
    let estimated_rows_per_partition = batch_size / num_output_partitions.max(1);
    let partition_buffers = (0..num_output_partitions)
        .map(|_| PartitionBuffer::new(schema.clone(), estimated_rows_per_partition))
        .collect();
```

- [ ] **Step 4: Implement scatter_batch**

Add the new method (see spec for full pseudocode). Key points:
- Takes `&RecordBatch` (borrow, not move) and `partition_ids: &[u32]` and `partition_starts: &[u32]`
- Column-oriented: iterates columns, then rows within each column
- For Fixed: access raw bytes via `column.to_data().buffers()[0].as_slice()`, copy `byte_width` bytes per row
- For Variable: downcast to StringArray/BinaryArray, access offsets and values
- For LargeVariable: downcast to LargeStringArray/LargeBinaryArray
- For Boolean: downcast to BooleanArray
- For Fallback: just store row index
- Null bits: `let is_valid = nulls.map_or(true, |n| n.is_valid(row));`
- Row counts from partition_starts (O(num_partitions))
- Auto-flush partitions that reached batch_size
- If has_fallback_columns, flush all non-empty partitions at end
- Precise memory tracking: before/after memory_size() comparison

- [ ] **Step 5: Add write_partition_batch helper**

```rust
fn write_partition_batch(&mut self, partition_id: usize, batch: &RecordBatch) -> Result<()> {
    self.partition_writers[partition_id].spill(
        &[batch.clone()],
        &self.runtime,
        &self.metrics,
        self.write_buffer_size,
        self.batch_size,
    )?;
    Ok(())
}
```

Note: this writes to the spill file. At shuffle_write time, spill file contents are copied to the output.

- [ ] **Step 6: Update partitioning_batch to call scatter_batch**

Replace all three `buffer_partitioned_batch_may_spill(input, ...)` calls with:
```rust
self.scatter_batch(&input, &scratch.partition_ids[..num_rows], &scratch.partition_starts)?;
```

Note: `scatter_batch` takes `&input` (borrow) not `input` (move). The `input` is no longer stored in `buffered_batches`. Also remove the `.await` since scatter_batch is not async.

- [ ] **Step 7: Delete old methods**

Delete: `buffer_partitioned_batch_may_spill`, `partitioned_batches`, `shuffle_write_partition`

- [ ] **Step 8: Rewrite spill()**

```rust
pub(crate) fn spill(&mut self) -> datafusion::common::Result<()> {
    let has_data = self.partition_buffers.iter().any(|b| b.row_count() > 0);
    if !has_data {
        return Ok(());
    }
    log::info!("ShuffleRepartitioner spilling...");
    with_trace("shuffle_spill", self.tracing_enabled, || {
        let mut spilled_bytes = 0;
        for p in 0..self.partition_buffers.len() {
            if self.partition_buffers[p].row_count() > 0 {
                let batch = self.partition_buffers[p].flush(None)?;
                spilled_bytes += self.partition_writers[p].spill(
                    &[batch], &self.runtime, &self.metrics,
                    self.write_buffer_size, self.batch_size)?;
            }
        }
        self.reservation.free();
        self.metrics.spill_count.add(1);
        self.metrics.spilled_bytes.add(spilled_bytes);
        Ok(())
    })
}
```

- [ ] **Step 9: Rewrite shuffle_write()**

Same structure as before but simpler — flush remaining partition buffers:

```rust
fn shuffle_write(&mut self) -> datafusion::common::Result<()> {
    with_trace("shuffle_write", self.tracing_enabled, || {
        let start_time = Instant::now();
        let num_output_partitions = self.partition_buffers.len();
        let mut offsets = vec![0; num_output_partitions + 1];

        let data_file = self.output_data_file.clone();
        let index_file = self.output_index_file.clone();

        let output_data = OpenOptions::new()
            .write(true).create(true).truncate(true).open(data_file)
            .map_err(|e| DataFusionError::Execution(format!("shuffle write error: {e:?}")))?;
        let mut output_data = BufWriter::new(output_data);

        #[allow(clippy::needless_range_loop)]
        for i in 0..num_output_partitions {
            offsets[i] = output_data.stream_position()?;

            // Copy spill file if any
            if let Some(spill_path) = self.partition_writers[i].path() {
                let mut spill_file = BufReader::new(File::open(spill_path)?);
                let mut wt = self.metrics.write_time.timer();
                std::io::copy(&mut spill_file, &mut output_data)?;
                wt.stop();
            }

            // Flush remaining in-memory buffer
            if self.partition_buffers[i].row_count() > 0 {
                let batch = self.partition_buffers[i].flush(None)?;
                let mut buf_batch_writer = BufBatchWriter::new(
                    &mut self.shuffle_block_writer, &mut output_data,
                    self.write_buffer_size, self.batch_size);
                buf_batch_writer.write(&batch,
                    &self.metrics.coalesce_time, &self.metrics.encode_time,
                    &self.metrics.write_time)?;
                buf_batch_writer.flush(
                    &self.metrics.coalesce_time, &self.metrics.encode_time,
                    &self.metrics.write_time)?;
            }
        }

        let mut wt = self.metrics.write_time.timer();
        output_data.flush()?;
        wt.stop();

        offsets[num_output_partitions] = output_data.stream_position()?;

        let mut wt = self.metrics.write_time.timer();
        let mut output_index = BufWriter::new(File::create(index_file)
            .map_err(|e| DataFusionError::Execution(format!("shuffle write error: {e:?}")))?);
        for offset in offsets {
            output_index.write_all(&(offset as i64).to_le_bytes()[..])?;
        }
        output_index.flush()?;
        wt.stop();

        self.metrics.baseline.elapsed_compute().add_duration(start_time.elapsed());
        Ok(())
    })
}
```

- [ ] **Step 10: Delete partitioned_batch_iterator.rs and update mod.rs**

```bash
rm native/core/src/execution/shuffle/partitioners/partitioned_batch_iterator.rs
```

In mod.rs, remove `mod partitioned_batch_iterator;` and `pub(super) use partitioned_batch_iterator::PartitionedBatchIterator;`.

- [ ] **Step 11: Build**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo check`
Fix any compile errors.

- [ ] **Step 12: Commit**

```bash
git add -u native/core/src/execution/shuffle/
git commit -m "feat: replace deferred-gather with scatter kernel for shuffle partitioning

Scatter values directly to per-partition typed buffers during insert,
eliminating interleave_record_batch (55% of shuffle write time) and
BatchCoalescer. Buffers auto-flush at batch_size, producing zero-copy
RecordBatches for the existing IPC+compress pipeline."
```

---

### Task 5: Run Rust tests and clippy

- [ ] **Step 1: Run tests**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet/native
DYLD_LIBRARY_PATH=$JAVA_HOME/lib/server cargo test
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets --workspace -- -D warnings
```

- [ ] **Step 3: Fix and commit any issues**

```bash
git add -u && git commit -m "fix: address test/clippy issues in scatter kernel"
```

---

### Task 6: Run JVM shuffle tests

- [ ] **Step 1: Build native**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && make core
```

- [ ] **Step 2: Run native shuffle suite**

```bash
./mvnw test -Dsuites="org.apache.comet.exec.CometNativeShuffleSuite"
```

- [ ] **Step 3: Run exec suite**

```bash
./mvnw test -Dsuites="org.apache.comet.exec.CometExecSuite"
```

- [ ] **Step 4: Fix any failures**

If failures occur, debug by adding logging to `scatter_batch` and comparing partition output with the old `interleave_record_batch` approach. Common issues:
- Null bitmap length mismatch: ensure `append_null_bit` is called for every row
- Offset mismatch in Variable: ensure initial offset `[0]` is not doubled
- Byte order: ensure fixed-width values use native endianness (Arrow stores native)

---

### Task 7: Format and cleanup

- [ ] **Step 1: Format**

```bash
cd /Users/andy/git/apache/temp/datafusion-comet && make format
```

- [ ] **Step 2: Check itertools usage**

```bash
grep -r "use itertools" native/core/src/
```

If only used in test code, keep it. If removed from all non-test code, consider removing from Cargo.toml.

- [ ] **Step 3: Commit**

```bash
git add -u && git commit -m "style: format code"
```
