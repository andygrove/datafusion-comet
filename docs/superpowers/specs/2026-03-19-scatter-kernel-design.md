# Custom Scatter Kernel for Shuffle Partitioning

## Problem

Comet's shuffle write path spends 55.4% of its time (702.9s out of 1268.3s in TPC-H 100GB) in `interleave_record_batch`, which gathers rows from buffered batches into per-partition output at write time. The buffered batches are no longer in CPU cache, causing cache-unfriendly random access. Arrow's `take_record_batch` (N calls per input batch, one per partition) has similar total cost because it still does O(rows) work per partition with intermediate RecordBatch allocations.

## Solution

Replace the deferred gather with a custom scatter kernel that appends values directly to per-partition typed buffers in a single pass during insert. This eliminates `interleave_record_batch`, `BatchCoalescer`, and all intermediate RecordBatch allocations.

### Scope

- First pass handles primitive types (boolean, integers, floats, decimal, date, timestamp) and string/binary types via typed column buffers.
- Complex types (struct, list, map, dictionary) fall back to row-index collection + `take_record_batch` at flush time.
- Multi-partition path only. `SinglePartitionShufflePartitioner` is unchanged.

## Data Flow

### Before (deferred gather)

```
insert_batch():
  compute partition_ids[row]             repart_time (74s, 5.8%)
  store (batch_id, row_id) indices       cheap
  buffer entire RecordBatch              cheap

shuffle_write():
  for each partition:
    interleave_record_batch()            gather_time (703s, 55.4%)
    BatchCoalescer                       coalesce_time (17s, 1.4%)
    Arrow IPC + compress                 encode_time (303s, 23.9%)
    write to disk                        write_time (76s, 6.0%)
```

### After (scatter during insert)

```
insert_batch():
  compute partition_ids[row]             repart_time (same)
  single pass: scatter values to partition_buffers[partition_id][col]
  when partition buffer reaches batch_size:
    flush -> RecordBatch (zero-copy) -> IPC + compress -> spill/output
  if memory pressure -> flush all partition buffers to spill files

shuffle_write():
  for each partition:
    copy spill file to output (if any)
    flush remaining buffer -> RecordBatch -> IPC + compress -> output
  write index file
```

The 703s gather step and 17s coalesce step are eliminated. No intermediate RecordBatch allocations during insert.

## PartitionBuffer Structure

New file: `native/core/src/execution/shuffle/partitioners/partition_buffer.rs`

```rust
pub(crate) struct PartitionBuffer {
    columns: Vec<ColumnBuffer>,
    row_count: usize,
    schema: SchemaRef,
}

pub(crate) enum ColumnBuffer {
    /// Bit-packed boolean values + null bitmap
    Boolean {
        values: BooleanBufferBuilder,
        nulls: BooleanBufferBuilder,
    },
    /// Fixed-width primitives: contiguous byte buffer + null bitmap
    /// Covers: Int8-64, UInt8-64, Float16/32/64, Date32/64,
    ///         Timestamp (all units), Decimal128
    Fixed {
        values: MutableBuffer,
        byte_width: usize,
        nulls: BooleanBufferBuilder,
    },
    /// Variable-width with i32 offsets: offsets + data bytes + null bitmap
    /// Covers: Utf8, Binary
    Variable {
        offsets: Vec<i32>,
        data: Vec<u8>,
        nulls: BooleanBufferBuilder,
    },
    /// Variable-width with i64 offsets: offsets + data bytes + null bitmap
    /// Covers: LargeUtf8, LargeBinary
    LargeVariable {
        offsets: Vec<i64>,
        data: Vec<u8>,
        nulls: BooleanBufferBuilder,
    },
    /// Unsupported types: collect row indices, use take at flush time
    /// Covers: Struct, List, Map, Dictionary, Decimal256, and any future types
    Fallback {
        indices: Vec<u32>,
    },
}
```

### Type Mapping

| Arrow Type | ColumnBuffer variant | byte_width |
|-----------|---------------------|-----------|
| Boolean | Boolean | N/A (bit-packed) |
| Int8, UInt8 | Fixed | 1 |
| Int16, UInt16, Float16 | Fixed | 2 |
| Int32, UInt32, Float32, Date32 | Fixed | 4 |
| Int64, UInt64, Float64, Date64, Timestamp(*) | Fixed | 8 |
| Decimal128 | Fixed | 16 |
| Utf8, Binary | Variable (i32 offsets) | N/A |
| LargeUtf8, LargeBinary | LargeVariable (i64 offsets) | N/A |
| Struct, List, Map, Dictionary, Decimal256 | Fallback | N/A |

### PartitionBuffer Methods

- `new(schema, batch_size, num_partitions)`: create column buffers based on schema types, pre-allocate for `batch_size / num_partitions` rows per partition (assuming uniform distribution)
- `memory_size() -> usize`: sum of all buffer capacities (exact tracking, includes Vec capacity and MutableBuffer capacity)
- `flush(fallback_batch: Option<&RecordBatch>) -> RecordBatch`: construct RecordBatch from buffers:
  - Fixed: `PrimitiveArray::new(ScalarBuffer::from(values), NullBuffer)` -- zero-copy
  - Variable: `StringArray::new(OffsetBuffer, Buffer, NullBuffer)` -- zero-copy
  - LargeVariable: `LargeStringArray::new(OffsetBuffer, Buffer, NullBuffer)` -- zero-copy
  - Boolean: `BooleanArray::new(BooleanBuffer, NullBuffer)` -- zero-copy
  - Fallback: `take_record_batch(fallback_batch, &indices)` for those columns only
  - Null count: derived from the null bitmap via `NullBuffer::new()` which computes it automatically
- `clear()`: reset all buffers and row_count for reuse after flush
- `has_fallback_columns() -> bool`: whether any column uses the Fallback variant

## Scatter Hot Path

In `partitioning_batch()`, after ScratchSpace computes `partition_ids` and `partition_starts`, scatter replaces `buffer_partitioned_batch_may_spill`. The approach is column-oriented: process one column at a time across all rows, which is better for CPU branch prediction (same code path for all rows of a column).

```rust
fn scatter_batch(
    &mut self,
    input: &RecordBatch,
    partition_ids: &[u32],
    partition_starts: &[u32],
) -> Result<()> {
    let num_rows = input.num_rows();
    let num_partitions = self.partition_buffers.len();

    // Compute per-partition row counts upfront from partition_starts
    // (already computed by ScratchSpace). O(num_partitions), not O(num_rows).
    let partition_counts: Vec<usize> = (0..num_partitions)
        .map(|p| (partition_starts[p + 1] - partition_starts[p]) as usize)
        .collect();

    // Track memory before scatter
    let mem_before: usize = self.partition_buffers.iter()
        .map(|b| b.memory_size()).sum();

    // Column-oriented scatter
    for (col_idx, column) in input.columns().iter().enumerate() {
        // Use concrete downcast to avoid to_data() overhead
        match &self.partition_buffers[0].columns[col_idx] {
            ColumnBuffer::Fixed { byte_width, .. } => {
                let byte_width = *byte_width;
                let data = column.to_data().buffers()[0].as_slice();
                let nulls = column.nulls();
                for row in 0..num_rows {
                    let p = partition_ids[row] as usize;
                    let src = &data[row * byte_width..(row + 1) * byte_width];
                    self.partition_buffers[p].columns[col_idx]
                        .append_fixed(src);
                    let is_valid = nulls.map_or(true, |n| n.is_valid(row));
                    self.partition_buffers[p].columns[col_idx]
                        .append_null_bit(is_valid);
                }
            }
            ColumnBuffer::Variable { .. } => {
                let arr = column.as_any().downcast_ref::<StringArray>()
                    .or_else(|| /* BinaryArray downcast */);
                let offsets = arr.offsets();
                let values = arr.values();
                let nulls = column.nulls();
                for row in 0..num_rows {
                    let p = partition_ids[row] as usize;
                    let start = offsets[row] as usize;
                    let end = offsets[row + 1] as usize;
                    self.partition_buffers[p].columns[col_idx]
                        .append_variable(&values[start..end]);
                    let is_valid = nulls.map_or(true, |n| n.is_valid(row));
                    self.partition_buffers[p].columns[col_idx]
                        .append_null_bit(is_valid);
                }
            }
            ColumnBuffer::LargeVariable { .. } => {
                // Same as Variable but with i64 offsets via LargeStringArray
                let arr = column.as_any()
                    .downcast_ref::<LargeStringArray>()
                    .or_else(|| /* LargeBinaryArray */);
                // ... same pattern with i64 offsets
            }
            ColumnBuffer::Boolean { .. } => {
                let arr = column.as_any()
                    .downcast_ref::<BooleanArray>().unwrap();
                let nulls = column.nulls();
                for row in 0..num_rows {
                    let p = partition_ids[row] as usize;
                    self.partition_buffers[p].columns[col_idx]
                        .append_bool(arr.value(row));
                    let is_valid = nulls.map_or(true, |n| n.is_valid(row));
                    self.partition_buffers[p].columns[col_idx]
                        .append_null_bit(is_valid);
                }
            }
            ColumnBuffer::Fallback { .. } => {
                for row in 0..num_rows {
                    let p = partition_ids[row] as usize;
                    self.partition_buffers[p].columns[col_idx]
                        .append_fallback_index(row as u32);
                }
            }
        }
    }

    // Update row counts from pre-computed partition_counts
    for p in 0..num_partitions {
        self.partition_buffers[p].row_count += partition_counts[p];
    }

    // Flush partitions that reached batch_size
    for p in 0..num_partitions {
        if self.partition_buffers[p].row_count >= self.batch_size {
            let batch = self.partition_buffers[p].flush(Some(input))?;
            self.write_partition_batch(p, &batch)?;
        }
    }

    // If schema has fallback columns, flush ALL partitions with pending
    // fallback indices, since those indices reference the current input
    // batch and will become stale when the next batch arrives.
    if self.has_fallback_columns {
        for p in 0..num_partitions {
            if self.partition_buffers[p].row_count > 0 {
                let batch = self.partition_buffers[p].flush(Some(input))?;
                self.write_partition_batch(p, &batch)?;
            }
        }
    }

    // Track memory precisely: compute delta from actual buffer sizes
    let mem_after: usize = self.partition_buffers.iter()
        .map(|b| b.memory_size()).sum();
    let mem_growth = mem_after.saturating_sub(mem_before);

    if self.reservation.try_grow(mem_growth).is_err() {
        self.spill()?;
    }
    Ok(())
}
```

### Key properties

- Column-oriented: one column at a time across all rows (branch prediction friendly)
- Inner loop for fixed-width is just a memcpy of `byte_width` bytes per row
- Null handling: when `null_count == 0`, append `true` to null bitmap unconditionally (ensures bitmap length matches row count). When `null_count > 0`, read from source null bitmap. This avoids a branch per row — the cost of appending a `true` bit is negligible.
- Fixed-width raw buffer access: use `column.to_data().buffers()[0].as_slice()` once per column (outside the row loop) to get the raw bytes. The `to_data()` cost is amortized over all rows. Alternatively, match on concrete types (`as_primitive::<Int32Type>()`) for each byte_width.
- Row counts computed from `partition_starts` in O(num_partitions), not O(num_rows)
- Flush check is O(num_partitions), not per-row
- Memory tracking is precise: compares actual buffer sizes before and after
- Fallback columns force flush every batch (correctness requirement since indices reference the current input)

## Spill Strategy

When memory pressure triggers a spill:

1. For each partition with `row_count > 0`:
   - `flush(None)` to get a RecordBatch. This is safe because: when `has_fallback_columns` is true, all Fallback indices were already flushed at the end of the preceding `scatter_batch` call; when `has_fallback_columns` is false, no Fallback columns exist.
   - Write through `PartitionWriter::spill()` (which uses BufBatchWriter -> IPC -> compress -> spill file)
   - `clear()` the buffer for reuse
2. Free the memory reservation
3. Continue accepting new batches -- partition writers have spill files, subsequent spills append

At final `shuffle_write()`:
- For each partition: copy spill file to output (if any), flush remaining buffer, write to output
- Write index file
- Same copy-on-write strategy as today

## Structural Changes

### New Files

- `native/core/src/execution/shuffle/partitioners/partition_buffer.rs`: `PartitionBuffer`, `ColumnBuffer`

### Remove

- `buffered_batches` and `partition_indices` from `MultiPartitionShuffleRepartitioner`
- `PartitionedBatchesProducer` and `PartitionedBatchIterator` (entire file `partitioned_batch_iterator.rs`)
- `interleave_record_batch` dependency
- `BatchCoalescer` usage in multi-partition path

### Modify

- `MultiPartitionShuffleRepartitioner`:
  - New field: `partition_buffers: Vec<PartitionBuffer>`
  - New field: `has_fallback_columns: bool` (computed once in try_new from schema)
  - Replace `buffer_partitioned_batch_may_spill` with `scatter_batch`
  - Rewrite `spill()` to flush partition buffers
  - Rewrite `shuffle_write()` to flush remaining buffers
- `PartitionWriter::spill()`: change signature from `&mut PartitionedBatchIterator` to `&[RecordBatch]`

### Keep Unchanged

- `ScratchSpace`
- `ShuffleBlockWriter` (codec.rs)
- `BufBatchWriter` (used inside PartitionWriter::spill for IPC encoding)
- `SinglePartitionShufflePartitioner`
- `ShuffleWriterExec`

## Fallback Strategy for Complex Types

When a schema has a mix of primitive and complex columns:
- Primitive columns use the scatter path (Fixed/Variable/Boolean buffers)
- Complex columns use `Fallback` which stores row indices
- At `flush()` time, Fallback columns are resolved by calling `take` on just those columns from the current input batch

When `has_fallback_columns` is true, all partition buffers are flushed at the end of every `scatter_batch` call to prevent stale index references. This means schemas with complex types cannot benefit from multi-batch buffering but still benefit from the scatter path for their primitive columns.

TPC-H uses only primitive and string types (Int32, Int64, Decimal128, Date32, Float64, Utf8) -- no columns hit the Fallback path.

## Testing

- Existing `CometNativeShuffleSuite` tests should pass unchanged (same output)
- Rust unit tests in `multi_partition.rs` need updating for new internal structure
- TPC-H before/after to measure improvement
- Spill tests with reduced memory budget
- Edge cases: empty partitions, single-row batches, all rows to one partition, all-null columns
- Mixed primitive + complex schemas to exercise Fallback path (not covered by TPC-H)
