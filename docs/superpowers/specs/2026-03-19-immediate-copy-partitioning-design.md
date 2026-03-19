# Immediate-Copy Shuffle Partitioning

## Problem

Comet's shuffle write path uses a deferred-gather partitioning strategy: during the insert phase, it stores `(batch_id, row_id)` indices per partition, buffering all input RecordBatches in memory. At write time, `interleave_record_batch` performs a random-access gather across all buffered batches to produce per-partition output. This gather step is the most expensive operation in the shuffle write path because the source batches are no longer in CPU cache when they're finally accessed.

## Solution

Replace deferred-gather with immediate-copy partitioning. When a batch arrives, immediately split it into per-partition sub-batches using Arrow's `take_record_batch` kernel and store them in per-partition `Vec<RecordBatch>` buffers. This eliminates the cross-batch gather step entirely and copies data while the source batch is hot in cache.

## Data Flow

### Before (deferred gather)

```
insert_batch(batch):
  compute partition IDs
  store (batch_id, row_id) in partition_indices[partition_id]
  buffered_batches.push(batch)
  if memory pressure -> spill (gather + write all partitions to spill files)

shuffle_write():
  for each partition:
    copy spill file to output (if any)
    PartitionedBatchIterator: interleave_record_batch(buffered_batches, indices)
    BufBatchWriter -> coalesce -> Arrow IPC -> compress -> disk
  write index file
```

### After (immediate copy)

```
insert_batch(batch):
  compute partition IDs
  group row indices by partition (ScratchSpace -- same as today)
  for each non-empty partition:
    small_batch = take_record_batch(&batch, &indices)
    partition_batches[partition_id].push(small_batch)
  track memory growth; if over budget -> spill()

shuffle_write():
  for each partition:
    copy spill file to output (if any)
    write remaining in-memory batches via BufBatchWriter -> coalesce -> IPC -> compress -> output
  write index file
```

## Structural Changes

### Remove

- `buffered_batches: Vec<RecordBatch>` from `MultiPartitionShuffleRepartitioner`
- `partition_indices: Vec<Vec<(u32, u32)>>` from `MultiPartitionShuffleRepartitioner`
- `PartitionedBatchesProducer` and `PartitionedBatchIterator` (entire file `partitioned_batch_iterator.rs`)
- The `interleave_record_batch` dependency

### Modify

**`MultiPartitionShuffleRepartitioner`** (`multi_partition.rs`):
- Replace `buffered_batches` and `partition_indices` with: `partition_batches: Vec<Vec<RecordBatch>>` -- one Vec per partition, accumulating `take` results during the insert phase
- `partitioning_batch()`: after computing partition indices via `ScratchSpace`, call `take_record_batch()` per non-empty partition, push result to `partition_batches[partition_id]`
- `spill()`: for each non-empty partition, create a temporary `BufBatchWriter` writing to the partition's spill file, write all accumulated batches, then clear the Vec. This matches how spill works today -- `BufBatchWriter` is created, used, and dropped during the spill operation.
- `shuffle_write()`: for each partition, copy spill file (if any) to output, then create `BufBatchWriter` for remaining in-memory batches, write to output, write index file
- Memory tracking: on each `take_record_batch` result, compute `batch.get_array_memory_size()` and call `reservation.try_grow()`. On spill, free the reservation.

**`PartitionWriter`** (`partition_writer.rs`):
- Change `spill()` signature from `&mut PartitionedBatchIterator` to `&[RecordBatch]` (or `impl Iterator<Item = Result<RecordBatch>>`). The body stays the same -- it creates a `BufBatchWriter`, writes batches through it, flushes.

### Keep Unchanged

- `ScratchSpace` -- still computes partition IDs and groups row indices by partition
- `BufBatchWriter` -- still does coalescing + buffered IPC writes (created temporarily during spill/write, not kept open)
- `ShuffleBlockWriter` (codec.rs) -- still does Arrow IPC + compression
- `SinglePartitionShufflePartitioner` -- already uses immediate writes
- `ShuffleWriterExec` -- just calls `insert_batch()` and `shuffle_write()`

## Hot Path: take per partition

```rust
// ScratchSpace already computed (same as today):
// - partition_starts[i]: where partition i's indices begin
// - partition_row_indices: flat array of row indices grouped by partition

for partition_id in 0..num_partitions {
    let start = scratch.partition_starts[partition_id] as usize;
    let end = scratch.partition_starts[partition_id + 1] as usize;
    if start == end {
        continue; // no rows for this partition
    }

    let indices = &scratch.partition_row_indices[start..end];
    let indices_array = UInt32Array::from_iter_values(indices.iter().copied());
    let partition_batch = take_record_batch(&batch, &indices_array)?;

    mem_growth += partition_batch.get_array_memory_size();
    self.partition_batches[partition_id].push(partition_batch);
}

// After processing all partitions for this batch, check memory
if self.reservation.try_grow(mem_growth).is_err() {
    self.spill()?;
}
```

Key properties:
- `ScratchSpace` already produces grouped indices -- used immediately instead of stored
- `take_record_batch` handles all Arrow types in one call
- Source batch is hot in cache since partition IDs were just computed from it
- Memory tracking is exact -- `get_array_memory_size()` on each take result
- Spill check happens after the current batch is fully partitioned, so no partial state

## Spill Strategy

When memory pressure triggers a spill:

1. For each partition with accumulated batches:
   - Call `partition_writer.spill(&partition_batches[partition_id])` which creates a temporary `BufBatchWriter` writing to the spill file, writes all batches through coalescer -> IPC -> compress pipeline, flushes and drops the writer
   - Clear `partition_batches[partition_id]`
2. Free the memory reservation
3. Continue accepting new batches -- partition writers have spill files, subsequent spills append

At final `shuffle_write()`:
- Partitions with spill files: copy spill bytes to output, then write remaining in-memory batches
- Partitions without spill files: write in-memory batches directly to output
- Same copy-on-write strategy as today

Memory tracking:
- Track cumulative `get_array_memory_size()` of all batches in all `partition_batches` Vecs
- On spill, free the full reservation (all Vecs are cleared)
- The `BufBatchWriter` is transient (created and dropped during spill/write), so its memory is short-lived and doesn't need reservation tracking

## Why Vec<RecordBatch> instead of per-partition BufBatchWriter

Keeping `BufBatchWriter` instances open during the entire insert phase would require them to write somewhere (`W: Write`). Writing to spill files during insert conflates normal buffering with spilling. Writing to in-memory `Vec<u8>` means tracking encoded IPC bytes (opaque) instead of RecordBatch sizes (transparent).

Using `Vec<RecordBatch>` per partition is simpler:
- Memory tracking is exact via `get_array_memory_size()`
- Spill creates `BufBatchWriter` temporarily, same as today
- No lifecycle management of long-lived writers
- Empty partitions are just empty Vecs (zero overhead for 2000+ partitions)
- `BufBatchWriter` + `BatchCoalescer` still coalesce small batches at write time

## Testing

- Existing `CometNativeShuffleSuite` tests (hash, range, round-robin, single partition, various data types) should pass unchanged -- same inputs, same outputs
- Rust unit tests in `multi_partition.rs` need updating for changed internal API
- Run TPC-H before/after to measure improvement
- Verify spilling with reduced memory budget
- Edge cases: empty partitions, single-row batches, all rows to one partition, many partitions with few rows
