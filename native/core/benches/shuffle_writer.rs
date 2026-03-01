// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::builder::{Date32Builder, Decimal128Builder, Int32Builder};
use arrow::array::{builder::StringBuilder, Array, Int32Array, RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, interleave_record_batch, take};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{RowConverter, SortField};
use comet::execution::shuffle::{
    CometPartitioning, CompressionCodec, ShuffleBlockWriter, ShuffleWriterExec,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_expr::expressions::{col, Column};
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::metrics::Time;
use datafusion::{
    physical_plan::{common::collect, ExecutionPlan},
    prelude::SessionContext,
};
use datafusion_comet_spark_expr::murmur3::create_murmur3_hashes;
use itertools::Itertools;
use std::io::Cursor;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Simulates hash partitioning to produce partition assignments for a batch.
/// Returns (partition_starts, partition_row_indices) just like ScratchSpace does.
fn compute_partition_assignments(
    batch: &RecordBatch,
    num_partitions: usize,
) -> (Vec<u32>, Vec<u32>) {
    let num_rows = batch.num_rows();
    let arrays = vec![Arc::clone(batch.column(0))];
    let mut hashes = vec![42u32; num_rows];
    create_murmur3_hashes(&arrays, &mut hashes).unwrap();

    let partition_ids: Vec<u32> = hashes
        .iter()
        .map(|h| h.rem_euclid(num_partitions as u32))
        .collect();

    // Count per partition
    let mut counts = vec![0u32; num_partitions + 1];
    for &pid in &partition_ids {
        counts[pid as usize] += 1;
    }

    // Accumulate into ends
    let mut accum = 0u32;
    for c in counts.iter_mut() {
        *c += accum;
        accum = *c;
    }

    // Build row indices (same algorithm as ScratchSpace)
    let mut row_indices = vec![0u32; num_rows];
    for (index, pid) in partition_ids.iter().enumerate().rev() {
        counts[*pid as usize] -= 1;
        row_indices[counts[*pid as usize] as usize] = index as u32;
    }

    (counts, row_indices)
}

/// Benchmark: the "take" approach — take from a single batch per partition
fn bench_take_partitioning(
    batch: &RecordBatch,
    partition_starts: &[u32],
    partition_row_indices: &[u32],
    num_partitions: usize,
) -> Vec<Vec<RecordBatch>> {
    let mut result: Vec<Vec<RecordBatch>> = vec![vec![]; num_partitions];
    for partition_id in 0..num_partitions {
        let start = partition_starts[partition_id] as usize;
        let end = partition_starts[partition_id + 1] as usize;
        if start == end {
            continue;
        }
        let indices = UInt32Array::from(partition_row_indices[start..end].to_vec());
        let columns: Vec<_> = batch
            .columns()
            .iter()
            .map(|col| take(col.as_ref(), &indices, None).unwrap())
            .collect();
        let sub_batch = RecordBatch::try_new(batch.schema(), columns).unwrap();
        result[partition_id].push(sub_batch);
    }
    result
}

/// Benchmark: the "take + coalesce" approach — take then concat when reaching batch_size
fn bench_take_coalesce_partitioning(
    batch: &RecordBatch,
    partition_starts: &[u32],
    partition_row_indices: &[u32],
    num_partitions: usize,
    batch_size: usize,
) -> Vec<Vec<RecordBatch>> {
    let mut result: Vec<Vec<RecordBatch>> = vec![vec![]; num_partitions];
    for partition_id in 0..num_partitions {
        let start = partition_starts[partition_id] as usize;
        let end = partition_starts[partition_id + 1] as usize;
        if start == end {
            continue;
        }
        let indices = UInt32Array::from(partition_row_indices[start..end].to_vec());
        let columns: Vec<_> = batch
            .columns()
            .iter()
            .map(|col| take(col.as_ref(), &indices, None).unwrap())
            .collect();
        let sub_batch = RecordBatch::try_new(batch.schema(), columns).unwrap();
        let partition = &mut result[partition_id];
        partition.push(sub_batch);

        let total_rows: usize = partition.iter().map(|b| b.num_rows()).sum();
        if total_rows >= batch_size {
            let schema = partition[0].schema();
            let coalesced = concat_batches(&schema, partition.iter()).unwrap();
            partition.clear();
            partition.push(coalesced);
        }
    }
    result
}

/// Benchmark: the old "interleave" approach — buffer batches + indices, then interleave
fn bench_interleave_partitioning(
    buffered_batches: &[RecordBatch],
    all_partition_indices: &[Vec<(u32, u32)>],
    num_partitions: usize,
    batch_size: usize,
) -> Vec<Vec<RecordBatch>> {
    let batch_refs: Vec<&RecordBatch> = buffered_batches.iter().collect();
    let mut result: Vec<Vec<RecordBatch>> = vec![vec![]; num_partitions];
    for partition_id in 0..num_partitions {
        let indices = &all_partition_indices[partition_id];
        if indices.is_empty() {
            continue;
        }
        let usize_indices: Vec<(usize, usize)> = indices
            .iter()
            .map(|(b, r)| (*b as usize, *r as usize))
            .collect();
        for chunk in usize_indices.chunks(batch_size) {
            let batch = interleave_record_batch(&batch_refs, chunk).unwrap();
            result[partition_id].push(batch);
        }
    }
    result
}

/// Micro-benchmarks comparing take vs interleave partitioning strategies
fn partitioning_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioning");

    for num_partitions in [16, 200] {
        let batch = create_batch(8192, true);
        let (partition_starts, partition_row_indices) =
            compute_partition_assignments(&batch, num_partitions);

        // Benchmark: take (single batch)
        group.bench_function(
            BenchmarkId::new("take", format!("{num_partitions}p")),
            |b| {
                b.iter(|| {
                    bench_take_partitioning(
                        &batch,
                        &partition_starts,
                        &partition_row_indices,
                        num_partitions,
                    )
                });
            },
        );

        // Benchmark: take + coalesce (single batch, but exercises the coalesce path)
        group.bench_function(
            BenchmarkId::new("take_coalesce", format!("{num_partitions}p")),
            |b| {
                b.iter(|| {
                    bench_take_coalesce_partitioning(
                        &batch,
                        &partition_starts,
                        &partition_row_indices,
                        num_partitions,
                        8192,
                    )
                });
            },
        );

        // Benchmark: simulate multi-batch take + coalesce (10 batches, like the end-to-end test)
        // This exercises the coalesce path more realistically since batches accumulate
        group.bench_function(
            BenchmarkId::new("take_coalesce_multi", format!("{num_partitions}p")),
            |b| {
                let batches = create_batches(8192, 10);
                // Pre-compute assignments for each batch
                let assignments: Vec<_> = batches
                    .iter()
                    .map(|batch| compute_partition_assignments(batch, num_partitions))
                    .collect();
                b.iter(|| {
                    let mut result: Vec<Vec<RecordBatch>> = vec![vec![]; num_partitions];
                    for (batch, (starts, indices)) in batches.iter().zip(&assignments) {
                        for partition_id in 0..num_partitions {
                            let start = starts[partition_id] as usize;
                            let end = starts[partition_id + 1] as usize;
                            if start == end {
                                continue;
                            }
                            let idx_array =
                                UInt32Array::from(indices[start..end].to_vec());
                            let columns: Vec<_> = batch
                                .columns()
                                .iter()
                                .map(|col| take(col.as_ref(), &idx_array, None).unwrap())
                                .collect();
                            let sub_batch =
                                RecordBatch::try_new(batch.schema(), columns).unwrap();
                            let partition = &mut result[partition_id];
                            partition.push(sub_batch);

                            let total_rows: usize =
                                partition.iter().map(|b| b.num_rows()).sum();
                            if total_rows >= 8192 {
                                let schema = partition[0].schema();
                                let coalesced =
                                    concat_batches(&schema, partition.iter()).unwrap();
                                partition.clear();
                                partition.push(coalesced);
                            }
                        }
                    }
                    result
                });
            },
        );

        // Benchmark: interleave (old approach) with equivalent multi-batch buffering
        group.bench_function(
            BenchmarkId::new("interleave_multi", format!("{num_partitions}p")),
            |b| {
                let batches = create_batches(8192, 10);
                // Pre-compute partition indices in the old (batch_idx, row_idx) format
                let mut all_indices: Vec<Vec<(u32, u32)>> = vec![vec![]; num_partitions];
                for (batch_idx, batch) in batches.iter().enumerate() {
                    let (starts, row_indices) =
                        compute_partition_assignments(batch, num_partitions);
                    for partition_id in 0..num_partitions {
                        let start = starts[partition_id] as usize;
                        let end = starts[partition_id + 1] as usize;
                        for &row_idx in &row_indices[start..end] {
                            all_indices[partition_id].push((batch_idx as u32, row_idx));
                        }
                    }
                }
                b.iter(|| {
                    bench_interleave_partitioning(
                        &batches,
                        &all_indices,
                        num_partitions,
                        8192,
                    )
                });
            },
        );
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let batch = create_batch(8192, true);
    let mut group = c.benchmark_group("shuffle_writer");
    for compression_codec in &[
        CompressionCodec::None,
        CompressionCodec::Lz4Frame,
        CompressionCodec::Snappy,
        CompressionCodec::Zstd(1),
        CompressionCodec::Zstd(6),
    ] {
        let name = format!("shuffle_writer: write encoded (compression={compression_codec:?})");
        group.bench_function(name, |b| {
            let mut buffer = vec![];
            let ipc_time = Time::default();
            let w =
                ShuffleBlockWriter::try_new(&batch.schema(), compression_codec.clone()).unwrap();
            b.iter(|| {
                buffer.clear();
                let mut cursor = Cursor::new(&mut buffer);
                w.write_batch(&batch, &mut cursor, &ipc_time).unwrap();
            });
        });
    }

    for compression_codec in [
        CompressionCodec::None,
        CompressionCodec::Lz4Frame,
        CompressionCodec::Snappy,
        CompressionCodec::Zstd(1),
        CompressionCodec::Zstd(6),
    ] {
        group.bench_function(
            format!("shuffle_writer: end to end (compression = {compression_codec:?})"),
            |b| {
                let ctx = SessionContext::new();
                let exec = create_shuffle_writer_exec(
                    compression_codec.clone(),
                    CometPartitioning::Hash(vec![Arc::new(Column::new("a", 0))], 16),
                );
                b.iter(|| {
                    let task_ctx = ctx.task_ctx();
                    let stream = exec.execute(0, task_ctx).unwrap();
                    let rt = Runtime::new().unwrap();
                    rt.block_on(collect(stream)).unwrap();
                });
            },
        );
    }

    let lex_ordering = LexOrdering::new(vec![PhysicalSortExpr::new_default(
        col("c0", batch.schema().as_ref()).unwrap(),
    )])
    .unwrap();

    let sort_fields: Vec<SortField> = batch
        .columns()
        .iter()
        .zip(&lex_ordering)
        .map(|(array, sort_expr)| {
            SortField::new_with_options(array.data_type().clone(), sort_expr.options)
        })
        .collect();
    let row_converter = RowConverter::new(sort_fields).unwrap();

    // These are hard-coded values based on the benchmark params of 8192 rows per batch, and 16
    // partitions. If these change, these values need to be recalculated, or bring over the
    // bounds-finding logic from shuffle_write_test in shuffle_writer.rs.
    let bounds_ints = vec![
        512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680,
    ];
    let bounds_array: Arc<dyn Array> = Arc::new(Int32Array::from(bounds_ints));
    let bounds_rows = row_converter
        .convert_columns(vec![bounds_array].as_slice())
        .unwrap();

    let owned_rows = bounds_rows.iter().map(|row| row.owned()).collect_vec();

    for partitioning in [
        CometPartitioning::Hash(vec![Arc::new(Column::new("a", 0))], 16),
        CometPartitioning::RangePartitioning(lex_ordering, 16, Arc::new(row_converter), owned_rows),
    ] {
        let compression_codec = CompressionCodec::None;
        group.bench_function(
            format!("shuffle_writer: end to end (partitioning={partitioning:?})"),
            |b| {
                let ctx = SessionContext::new();
                let exec =
                    create_shuffle_writer_exec(compression_codec.clone(), partitioning.clone());
                b.iter(|| {
                    let task_ctx = ctx.task_ctx();
                    let stream = exec.execute(0, task_ctx).unwrap();
                    let rt = Runtime::new().unwrap();
                    rt.block_on(collect(stream)).unwrap();
                });
            },
        );
    }

    // Benchmark with varying partition counts to isolate partitioning overhead
    for num_partitions in [16, 200] {
        let compression_codec = CompressionCodec::None;
        let partitioning =
            CometPartitioning::Hash(vec![Arc::new(Column::new("a", 0))], num_partitions);
        group.bench_function(
            BenchmarkId::new("end_to_end_hash", format!("{num_partitions}p")),
            |b| {
                let ctx = SessionContext::new();
                let exec =
                    create_shuffle_writer_exec(compression_codec.clone(), partitioning.clone());
                b.iter(|| {
                    let task_ctx = ctx.task_ctx();
                    let stream = exec.execute(0, task_ctx).unwrap();
                    let rt = Runtime::new().unwrap();
                    rt.block_on(collect(stream)).unwrap();
                });
            },
        );
    }
}

fn create_shuffle_writer_exec(
    compression_codec: CompressionCodec,
    partitioning: CometPartitioning,
) -> ShuffleWriterExec {
    let batches = create_batches(8192, 10);
    let schema = batches[0].schema();
    let partitions = &[batches];
    ShuffleWriterExec::try_new(
        Arc::new(DataSourceExec::new(Arc::new(
            MemorySourceConfig::try_new(partitions, Arc::clone(&schema), None).unwrap(),
        ))),
        partitioning,
        compression_codec,
        "/tmp/data.out".to_string(),
        "/tmp/index.out".to_string(),
        false,
        1024 * 1024,
    )
    .unwrap()
}

fn create_batches(size: usize, count: usize) -> Vec<RecordBatch> {
    let batch = create_batch(size, true);
    let mut batches = Vec::new();
    for _ in 0..count {
        batches.push(batch.clone());
    }
    batches
}

fn create_batch(num_rows: usize, allow_nulls: bool) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c0", DataType::Int32, true),
        Field::new("c1", DataType::Utf8, true),
        Field::new("c2", DataType::Date32, true),
        Field::new("c3", DataType::Decimal128(11, 2), true),
    ]));
    let mut a = Int32Builder::new();
    let mut b = StringBuilder::new();
    let mut c = Date32Builder::new();
    let mut d = Decimal128Builder::new()
        .with_precision_and_scale(11, 2)
        .unwrap();
    for i in 0..num_rows {
        a.append_value(i as i32);
        c.append_value(i as i32);
        d.append_value((i * 1000000) as i128);
        if allow_nulls && i % 10 == 0 {
            b.append_null();
        } else {
            b.append_value(format!("this is string number {i}"));
        }
    }
    let a = a.finish();
    let b = b.finish();
    let c = c.finish();
    let d = d.finish();
    RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(a), Arc::new(b), Arc::new(c), Arc::new(d)],
    )
    .unwrap()
}

fn config() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = config();
    targets = criterion_benchmark, partitioning_benchmark
}
criterion_main!(benches);
