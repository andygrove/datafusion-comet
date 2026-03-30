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

use crate::metrics::ShufflePartitionerMetrics;
use crate::partitioners::ShufflePartitioner;
use crate::writers::PartitionWriter;
use crate::{comet_partitioning, CometPartitioning, CompressionCodec, ShuffleBlockWriter};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::interleave_record_batch;
use arrow::datatypes::SchemaRef;
use datafusion::common::DataFusionError;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion_comet_common::tracing::{with_trace, with_trace_async};
use datafusion_comet_spark_expr::murmur3::create_murmur3_hashes;
use itertools::Itertools;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use std::sync::Arc;
use tokio::time::Instant;

/// Reusable scratch buffers for computing row-to-partition assignments.
#[derive(Default)]
struct ScratchSpace {
    /// Hashes for each row in the current batch.
    hashes_buf: Vec<u32>,
    /// Partition ids for each row in the current batch.
    partition_ids: Vec<u32>,
    /// The row indices of the rows in each partition. This array is conceptually divided into
    /// partitions, where each partition contains the row indices of the rows in that partition.
    /// The length of this array is the same as the number of rows in the batch.
    partition_row_indices: Vec<u32>,
    /// The start indices of partitions in partition_row_indices. partition_starts[K] and
    /// partition_starts[K + 1] are the start and end indices of partition K in partition_row_indices.
    /// The length of this array is 1 + the number of partitions.
    partition_starts: Vec<u32>,
    /// Reusable buffer for interleave_record_batch indices, avoids per-partition allocation.
    interleave_indices: Vec<(usize, usize)>,
    /// Whether hashes_buf is used by the current partitioning scheme.
    uses_hashes: bool,
}

impl ScratchSpace {
    /// Ensure scratch buffers can handle the given number of rows.
    fn ensure_capacity(&mut self, num_rows: usize) {
        if num_rows > self.partition_ids.len() {
            self.partition_ids.resize(num_rows, 0);
        }
        if self.uses_hashes && num_rows > self.hashes_buf.len() {
            self.hashes_buf.resize(num_rows, 0);
        }
    }

    fn map_partition_ids_to_starts_and_indices(
        &mut self,
        num_output_partitions: usize,
        num_rows: usize,
    ) {
        let partition_ids = &mut self.partition_ids[..num_rows];

        // count each partition size, while leaving the last extra element as 0
        let partition_counters = &mut self.partition_starts;
        partition_counters.resize(num_output_partitions + 1, 0);
        partition_counters.fill(0);
        partition_ids
            .iter()
            .for_each(|partition_id| partition_counters[*partition_id as usize] += 1);

        // accumulate partition counters into partition ends
        // e.g. partition counter: [1, 3, 2, 1, 0] => [1, 4, 6, 7, 7]
        let partition_ends = partition_counters;
        let mut accum = 0;
        partition_ends.iter_mut().for_each(|v| {
            *v += accum;
            accum = *v;
        });

        // calculate partition row indices and partition starts
        // e.g. partition ids: [3, 1, 1, 1, 2, 2, 0] will produce the following partition_row_indices
        // and partition_starts arrays:
        //
        //  partition_row_indices: [6, 1, 2, 3, 4, 5, 0]
        //  partition_starts: [0, 1, 4, 6, 7]
        //
        // partition_starts conceptually splits partition_row_indices into smaller slices.
        // Each slice partition_row_indices[partition_starts[K]..partition_starts[K + 1]] contains the
        // row indices of the input batch that are partitioned into partition K. For example,
        // first partition 0 has one row index [6], partition 1 has row indices [1, 2, 3], etc.
        let partition_row_indices = &mut self.partition_row_indices;
        partition_row_indices.resize(num_rows, 0);
        for (index, partition_id) in partition_ids.iter().enumerate().rev() {
            partition_ends[*partition_id as usize] -= 1;
            let end = partition_ends[*partition_id as usize];
            partition_row_indices[end as usize] = index as u32;
        }
    }
}

/// A partitioner that uses a hash function to partition data into multiple partitions.
///
/// Each input batch is immediately repartitioned and serialized to compressed IPC blocks
/// in per-partition memory buffers. When the memory pool limit is reached, buffers are
/// spilled to disk as compressed IPC data, avoiding the need to buffer uncompressed
/// record batches.
pub(crate) struct MultiPartitionShuffleRepartitioner {
    output_data_file: String,
    output_index_file: String,
    partition_writers: Vec<PartitionWriter>,
    /// Partitioning scheme to use
    partitioning: CometPartitioning,
    runtime: Arc<RuntimeEnv>,
    metrics: ShufflePartitionerMetrics,
    /// Reused scratch space for computing partition indices
    scratch: ScratchSpace,
    /// Reservation for repartitioning
    reservation: MemoryReservation,
    tracing_enabled: bool,
}

impl MultiPartitionShuffleRepartitioner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        partition: usize,
        output_data_file: String,
        output_index_file: String,
        schema: SchemaRef,
        partitioning: CometPartitioning,
        metrics: ShufflePartitionerMetrics,
        runtime: Arc<RuntimeEnv>,
        batch_size: usize,
        codec: CompressionCodec,
        tracing_enabled: bool,
        _write_buffer_size: usize,
    ) -> datafusion::common::Result<Self> {
        let num_output_partitions = partitioning.partition_count();
        assert_ne!(
            num_output_partitions, 1,
            "Use SinglePartitionShufflePartitioner for 1 output partition."
        );

        let uses_hashes = matches!(
            partitioning,
            CometPartitioning::Hash(_, _) | CometPartitioning::RoundRobin(_, _)
        );

        let scratch = ScratchSpace {
            hashes_buf: if uses_hashes {
                vec![0; batch_size]
            } else {
                vec![]
            },
            partition_ids: vec![0; batch_size],
            partition_row_indices: vec![0; batch_size],
            partition_starts: vec![0; num_output_partitions + 1],
            interleave_indices: vec![],
            uses_hashes,
        };

        let shuffle_block_writer = ShuffleBlockWriter::try_new(schema.as_ref(), codec.clone())?;

        let partition_writers = (0..num_output_partitions)
            .map(|_| PartitionWriter::new(shuffle_block_writer.clone(), batch_size))
            .collect::<Vec<_>>();

        let reservation = MemoryConsumer::new(format!("ShuffleRepartitioner[{partition}]"))
            .with_can_spill(true)
            .register(&runtime.memory_pool);

        Ok(Self {
            output_data_file,
            output_index_file,
            partition_writers,
            partitioning,
            runtime,
            metrics,
            scratch,
            reservation,
            tracing_enabled,
        })
    }

    /// Shuffles rows in input batch into corresponding partition buffer.
    /// This function first calculates hashes for rows and then immediately repartitions
    /// and writes the rows to per-partition buffers as compressed IPC blocks.
    /// This should not be called directly. Use `insert_batch` instead.
    fn partitioning_batch(&mut self, input: &RecordBatch) -> datafusion::common::Result<()> {
        if input.num_rows() == 0 {
            return Ok(());
        }

        // Update data size metric
        self.metrics.data_size.add(input.get_array_memory_size());

        // NOTE: in shuffle writer exec, the output_rows metrics represents the
        // number of rows those are written to output data file.
        self.metrics.baseline.record_output(input.num_rows());

        match &self.partitioning {
            CometPartitioning::Hash(exprs, num_output_partitions) => {
                let mut scratch = std::mem::take(&mut self.scratch);
                scratch.ensure_capacity(input.num_rows());
                let (partition_starts, partition_row_indices): (&Vec<u32>, &Vec<u32>) = {
                    let mut timer = self.metrics.repart_time.timer();

                    // Evaluate partition expressions to get rows to apply partitioning scheme.
                    let arrays = exprs
                        .iter()
                        .map(|expr| expr.evaluate(input)?.into_array(input.num_rows()))
                        .collect::<datafusion::common::Result<Vec<_>>>()?;

                    let num_rows = arrays[0].len();

                    // Use identical seed as Spark hash partitioning.
                    let hashes_buf = &mut scratch.hashes_buf[..num_rows];
                    hashes_buf.fill(42_u32);

                    // Generate partition ids for every row.
                    {
                        // Hash arrays and compute partition ids based on number of partitions.
                        let partition_ids = &mut scratch.partition_ids[..num_rows];
                        create_murmur3_hashes(&arrays, hashes_buf)?
                            .iter()
                            .enumerate()
                            .for_each(|(idx, hash)| {
                                partition_ids[idx] =
                                    comet_partitioning::pmod(*hash, *num_output_partitions) as u32;
                            });
                    }

                    // We now have partition ids for every input row, map that to partition starts
                    // and partition indices to eventually right these rows to partition buffers.
                    scratch
                        .map_partition_ids_to_starts_and_indices(*num_output_partitions, num_rows);

                    timer.stop();
                    Ok::<(&Vec<u32>, &Vec<u32>), DataFusionError>((
                        &scratch.partition_starts,
                        &scratch.partition_row_indices,
                    ))
                }?;

                self.repartition_and_write(
                    input,
                    partition_row_indices,
                    partition_starts,
                    &mut scratch.interleave_indices,
                )?;
                self.scratch = scratch;
            }
            CometPartitioning::RangePartitioning(
                lex_ordering,
                num_output_partitions,
                row_converter,
                bounds,
            ) => {
                let mut scratch = std::mem::take(&mut self.scratch);
                scratch.ensure_capacity(input.num_rows());
                let (partition_starts, partition_row_indices): (&Vec<u32>, &Vec<u32>) = {
                    let mut timer = self.metrics.repart_time.timer();

                    // Evaluate partition expressions for values to apply partitioning scheme on.
                    let arrays = lex_ordering
                        .iter()
                        .map(|expr| expr.expr.evaluate(input)?.into_array(input.num_rows()))
                        .collect::<datafusion::common::Result<Vec<_>>>()?;

                    let num_rows = arrays[0].len();

                    // Generate partition ids for every row, first by converting the partition
                    // arrays to Rows, and then doing binary search for each Row against the
                    // bounds Rows.
                    {
                        let row_batch = row_converter.convert_columns(arrays.as_slice())?;
                        let partition_ids = &mut scratch.partition_ids[..num_rows];

                        row_batch.iter().enumerate().for_each(|(row_idx, row)| {
                            partition_ids[row_idx] = bounds
                                .as_slice()
                                .partition_point(|bound| bound.row() <= row)
                                as u32
                        });
                    }

                    // We now have partition ids for every input row, map that to partition starts
                    // and partition indices to eventually right these rows to partition buffers.
                    scratch
                        .map_partition_ids_to_starts_and_indices(*num_output_partitions, num_rows);

                    timer.stop();
                    Ok::<(&Vec<u32>, &Vec<u32>), DataFusionError>((
                        &scratch.partition_starts,
                        &scratch.partition_row_indices,
                    ))
                }?;

                self.repartition_and_write(
                    input,
                    partition_row_indices,
                    partition_starts,
                    &mut scratch.interleave_indices,
                )?;
                self.scratch = scratch;
            }
            CometPartitioning::RoundRobin(num_output_partitions, max_hash_columns) => {
                // Comet implements "round robin" as hash partitioning on columns.
                // This achieves the same goal as Spark's round robin (even distribution
                // without semantic grouping) while being deterministic for fault tolerance.
                //
                // Note: This produces different partition assignments than Spark's round robin,
                // which sorts by UnsafeRow binary representation before assigning partitions.
                // However, both approaches provide even distribution and determinism.
                let mut scratch = std::mem::take(&mut self.scratch);
                scratch.ensure_capacity(input.num_rows());
                let (partition_starts, partition_row_indices): (&Vec<u32>, &Vec<u32>) = {
                    let mut timer = self.metrics.repart_time.timer();

                    let num_rows = input.num_rows();

                    // Collect columns for hashing, respecting max_hash_columns limit
                    // max_hash_columns of 0 means no limit (hash all columns)
                    // Negative values are normalized to 0 in the planner
                    let num_columns_to_hash = if *max_hash_columns == 0 {
                        input.num_columns()
                    } else {
                        (*max_hash_columns).min(input.num_columns())
                    };
                    let columns_to_hash: Vec<ArrayRef> = (0..num_columns_to_hash)
                        .map(|i| Arc::clone(input.column(i)))
                        .collect();

                    // Use identical seed as Spark hash partitioning.
                    let hashes_buf = &mut scratch.hashes_buf[..num_rows];
                    hashes_buf.fill(42_u32);

                    // Compute hash for selected columns
                    create_murmur3_hashes(&columns_to_hash, hashes_buf)?;

                    // Assign partition IDs based on hash (same as hash partitioning)
                    let partition_ids = &mut scratch.partition_ids[..num_rows];
                    hashes_buf.iter().enumerate().for_each(|(idx, hash)| {
                        partition_ids[idx] =
                            comet_partitioning::pmod(*hash, *num_output_partitions) as u32;
                    });

                    // We now have partition ids for every input row, map that to partition starts
                    // and partition indices to eventually write these rows to partition buffers.
                    scratch
                        .map_partition_ids_to_starts_and_indices(*num_output_partitions, num_rows);

                    timer.stop();
                    Ok::<(&Vec<u32>, &Vec<u32>), DataFusionError>((
                        &scratch.partition_starts,
                        &scratch.partition_row_indices,
                    ))
                }?;

                self.repartition_and_write(
                    input,
                    partition_row_indices,
                    partition_starts,
                    &mut scratch.interleave_indices,
                )?;
                self.scratch = scratch;
            }
            other => {
                // this should be unreachable as long as the validation logic
                // in the constructor is kept up-to-date
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported shuffle partitioning scheme {other:?}"
                )));
            }
        }
        Ok(())
    }

    /// Immediately repartition the input batch and write each partition's rows to the
    /// corresponding partition writer as compressed IPC blocks. Checks memory after each
    /// partition write and spills all partitions if the pool limit is exceeded.
    fn repartition_and_write(
        &mut self,
        input: &RecordBatch,
        partition_row_indices: &[u32],
        partition_starts: &[u32],
        interleave_indices: &mut Vec<(usize, usize)>,
    ) -> datafusion::common::Result<()> {
        let batch_refs = vec![input];

        for (partition_id, (&start, &end)) in partition_starts
            .iter()
            .tuple_windows()
            .enumerate()
            .filter(|(_, (start, end))| start < end)
        {
            let row_indices = &partition_row_indices[start as usize..end as usize];
            interleave_indices.clear();
            interleave_indices.extend(row_indices.iter().map(|&r| (0, r as usize)));

            let partition_batch = {
                let mut interleave_timer = self.metrics.interleave_time.timer();
                let batch =
                    interleave_record_batch(&batch_refs, interleave_indices).map_err(|e| {
                        DataFusionError::ArrowError(
                            Box::from(e),
                            Some(DataFusionError::get_back_trace()),
                        )
                    })?;
                interleave_timer.stop();
                batch
            };

            let before = self.partition_writers[partition_id].buffered_bytes();
            self.partition_writers[partition_id].write(
                &partition_batch,
                &self.metrics.encode_time,
                &self.metrics.write_time,
                &self.metrics.coalesce_time,
            )?;
            let after = self.partition_writers[partition_id].buffered_bytes();

            if after > before {
                if self.reservation.try_grow(after - before).is_err() {
                    self.spill()?;
                }
            } else {
                // Compression reduced buffered size — return memory to the pool
                self.reservation.shrink(before - after);
            }
        }

        Ok(())
    }

    fn used(&self) -> usize {
        self.reservation.size()
    }

    fn spilled_bytes(&self) -> usize {
        self.metrics.spilled_bytes.value()
    }

    fn spill_count(&self) -> usize {
        self.metrics.spill_count.value()
    }

    fn data_size(&self) -> usize {
        self.metrics.data_size.value()
    }

    pub(crate) fn spill(&mut self) -> datafusion::common::Result<()> {
        log::info!(
            "ShuffleRepartitioner spilling shuffle data of {} to disk while inserting ({} time(s) so far)",
            self.used(),
            self.spill_count()
        );

        with_trace("shuffle_spill", self.tracing_enabled, || {
            let mut spilled_bytes = 0;

            for partition_writer in &mut self.partition_writers {
                spilled_bytes += partition_writer.spill_buffer(
                    &self.runtime,
                    &self.metrics.encode_time,
                    &self.metrics.write_time,
                    &self.metrics.coalesce_time,
                )?;
            }

            self.reservation.free();
            self.metrics.spill_count.add(1);
            self.metrics.spilled_bytes.add(spilled_bytes);
            Ok(())
        })
    }

    #[cfg(test)]
    pub(crate) fn partition_writers(&self) -> &[PartitionWriter] {
        &self.partition_writers
    }
}

#[async_trait::async_trait]
impl ShufflePartitioner for MultiPartitionShuffleRepartitioner {
    /// Shuffles rows in input batch into corresponding partition buffers.
    async fn insert_batch(&mut self, batch: RecordBatch) -> datafusion::common::Result<()> {
        with_trace_async("shuffle_insert_batch", self.tracing_enabled, || async {
            let start_time = Instant::now();
            self.partitioning_batch(&batch)?;
            self.metrics.input_batches.add(1);
            self.metrics
                .baseline
                .elapsed_compute()
                .add_duration(start_time.elapsed());
            Ok(())
        })
        .await
    }

    /// Writes buffered shuffled record batches into Arrow IPC bytes.
    fn shuffle_write(&mut self) -> datafusion::common::Result<()> {
        with_trace("shuffle_write", self.tracing_enabled, || {
            let start_time = Instant::now();

            let num_output_partitions = self.partition_writers.len();
            let mut offsets = vec![0; num_output_partitions + 1];

            let data_file = self.output_data_file.clone();
            let index_file = self.output_index_file.clone();

            let output_data = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(data_file)
                .map_err(|e| DataFusionError::Execution(format!("shuffle write error: {e:?}")))?;

            let mut output_data = BufWriter::new(output_data);

            #[allow(clippy::needless_range_loop)]
            for i in 0..num_output_partitions {
                offsets[i] = output_data.stream_position()?;

                // Copy spill file contents if any
                if let Some(spill_path) = self.partition_writers[i].path() {
                    let mut spill_file = File::open(spill_path)?;
                    let mut write_timer = self.metrics.write_time.timer();
                    std::io::copy(&mut spill_file, &mut output_data)?;
                    write_timer.stop();
                }

                // Write remaining in-memory buffer
                let buffer = self.partition_writers[i].drain_buffer(
                    &self.metrics.encode_time,
                    &self.metrics.write_time,
                    &self.metrics.coalesce_time,
                )?;
                if !buffer.is_empty() {
                    let mut write_timer = self.metrics.write_time.timer();
                    output_data.write_all(&buffer)?;
                    write_timer.stop();
                }
            }

            let mut write_timer = self.metrics.write_time.timer();
            output_data.flush()?;
            write_timer.stop();

            // add one extra offset at last to ease partition length computation
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
}

impl Debug for MultiPartitionShuffleRepartitioner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShuffleRepartitioner")
            .field("memory_used", &self.used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spilled_count", &self.spill_count())
            .field("data_size", &self.data_size())
            .finish()
    }
}
