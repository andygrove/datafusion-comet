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

//! Defines the External shuffle repartition plan.

use crate::{
    common::bit::ceil,
    errors::{CometError, CometResult},
};
use arrow::compute::{concat_batches, filter_record_batch};
use arrow::{datatypes::*, ipc::writer::StreamWriter};
use async_trait::async_trait;
use bytes::Buf;
use crc32fast::Hasher;
use datafusion::{
    arrow::{
        array::*,
        datatypes::{DataType, SchemaRef, TimeUnit},
        error::ArrowError,
        record_batch::RecordBatch,
    },
    error::{DataFusionError, Result},
    execution::{
        context::TaskContext,
        disk_manager::RefCountedTempFile,
        memory_pool::{MemoryConsumer, MemoryReservation},
        runtime_env::RuntimeEnv,
    },
    physical_plan::{
        metrics::{
            BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, Time,
        },
        stream::RecordBatchStreamAdapter,
        DisplayAs, DisplayFormatType, ExecutionMode, ExecutionPlan, Partitioning, PlanProperties,
        RecordBatchStream, SendableRecordBatchStream, Statistics,
    },
};
use datafusion_comet_spark_expr::spark_hash::create_murmur3_hashes;
use datafusion_physical_expr::EquivalenceProperties;
use futures::executor::block_on;
use futures::{lock::Mutex, Stream, StreamExt, TryFutureExt, TryStreamExt};
use itertools::Itertools;
use simd_adler32::Adler32;
use std::io::Error;
use std::{
    any::Any,
    fmt,
    fmt::{Debug, Formatter},
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Arc,
    task::{Context, Poll},
};
use tokio::time::Instant;

/// The shuffle writer operator maps each input partition to M output partitions based on a
/// partitioning scheme. No guarantees are made about the order of the resulting partitions.
#[derive(Debug)]
pub struct ShuffleWriterExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Partitioning scheme to use
    partitioning: Partitioning,
    /// Output data file path
    output_data_file: String,
    /// Output index file path
    output_index_file: String,
    /// Metrics
    metrics: ExecutionPlanMetricsSet,
    cache: PlanProperties,
}

impl DisplayAs for ShuffleWriterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "ShuffleWriterExec: partitioning={:?}", self.partitioning)
            }
        }
    }
}

#[async_trait]
impl ExecutionPlan for ShuffleWriterExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(ShuffleWriterExec::try_new(
                Arc::clone(&children[0]),
                self.partitioning.clone(),
                self.output_data_file.clone(),
                self.output_index_file.clone(),
            )?)),
            _ => panic!("ShuffleWriterExec wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let metrics = ShuffleRepartitionerMetrics::new(&self.metrics, 0);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            futures::stream::once(
                external_shuffle(
                    input,
                    partition,
                    self.output_data_file.clone(),
                    self.output_index_file.clone(),
                    self.partitioning.clone(),
                    metrics,
                    context,
                )
                .map_err(|e| ArrowError::ExternalError(Box::new(e))),
            )
            .try_flatten(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        self.input.statistics()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn name(&self) -> &str {
        "ShuffleWriterExec"
    }
}

impl ShuffleWriterExec {
    /// Create a new ShuffleWriterExec
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partitioning: Partitioning,
        output_data_file: String,
        output_index_file: String,
    ) -> Result<Self> {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&input.schema())),
            partitioning.clone(),
            ExecutionMode::Bounded,
        );

        Ok(ShuffleWriterExec {
            input,
            partitioning,
            metrics: ExecutionPlanMetricsSet::new(),
            output_data_file,
            output_index_file,
            cache,
        })
    }
}

struct PartitionBuffer {
    /// The schema of batches to be partitioned.
    schema: SchemaRef,
    /// The "frozen" Arrow IPC bytes of active data. They are frozen when `flush` is called.
    frozen: Vec<u8>,
    /// Array builders for appending rows into buffering batches.
    active: Vec<RecordBatch>,
    /// The estimation of memory size of active builders in bytes when they are filled.
    active_slots_mem_size: usize,
    /// Number of rows in active builders.
    num_active_rows: usize,
    /// The maximum number of rows in a batch. Once `num_active_rows` reaches `batch_size`,
    /// the active array builders will be frozen and appended to frozen buffer `frozen`.
    batch_size: usize,
    /// Memory reservation for this partition buffer.
    reservation: MemoryReservation,
}

impl PartitionBuffer {
    fn new(
        schema: SchemaRef,
        batch_size: usize,
        partition_id: usize,
        runtime: &Arc<RuntimeEnv>,
    ) -> Self {
        let reservation = MemoryConsumer::new(format!("PartitionBuffer[{}]", partition_id))
            .with_can_spill(true)
            .register(&runtime.memory_pool);

        Self {
            schema,
            frozen: vec![],
            active: vec![],
            active_slots_mem_size: 0,
            num_active_rows: 0,
            batch_size,
            reservation,
        }
    }

    /// Initializes active builders if necessary.
    /// Returns error if memory reservation fails.
    fn init_active_if_necessary(&mut self, metrics: &ShuffleRepartitionerMetrics) -> Result<isize> {
        let mut mem_diff = 0;
        // TODO reinstate this logic

        // if self.active.is_none() {
        //     // Estimate the memory size of active builders
        //     if self.active_slots_mem_size == 0 {
        //         self.active_slots_mem_size = self
        //             .schema
        //             .fields()
        //             .iter()
        //             .map(|field| slot_size(self.batch_size, field.data_type()))
        //             .sum::<usize>();
        //     }
        //
        //     let mut mempool_timer = metrics.mempool_time.timer();
        //     self.reservation.try_grow(self.active_slots_mem_size)?;
        //     mempool_timer.stop();
        //
        //     let mut repart_timer = metrics.repart_time.timer();
        //     self.active = new_array_builders(&self.schema, self.batch_size);
        //     repart_timer.stop();
        //
        //     mem_diff += self.active_slots_mem_size as isize;
        // }
        Ok(mem_diff)
    }

    /// flush active data into frozen bytes
    fn flush(&mut self, ipc_time: &Time) -> Result<isize> {
        if self.num_active_rows == 0 {
            return Ok(0);
        }
        let mut mem_diff = 0isize;

        // active -> staging
        self.num_active_rows = 0;
        self.reservation.try_shrink(self.active_slots_mem_size)?;

        let frozen_capacity_old = self.frozen.capacity();
        let frozen_batches = std::mem::take(&mut self.active);

        let first_batch = &frozen_batches[0];
        let mut all_same_schema = true;
        for i in 0..frozen_batches.len() {
            if first_batch.schema() != frozen_batches[i].schema() {
                all_same_schema = false;
                break;
            }
        }
        let mut cursor = Cursor::new(&mut self.frozen);
        if all_same_schema {
            let batch = concat_batches(&first_batch.schema(), frozen_batches.iter().collect_vec())?;
            cursor.seek(SeekFrom::End(0))?;
            write_ipc_compressed(&batch, &mut cursor, ipc_time)?;
        } else {
            for frozen_batch in frozen_batches {
                cursor.seek(SeekFrom::End(0))?;
                write_ipc_compressed(&frozen_batch, &mut cursor, ipc_time)?;
            }
        }
        mem_diff += (self.frozen.capacity() - frozen_capacity_old) as isize;
        Ok(mem_diff)
    }
}

fn slot_size(len: usize, data_type: &DataType) -> usize {
    match data_type {
        DataType::Boolean => ceil(len, 8),
        DataType::Int8 => len,
        DataType::Int16 => len * 2,
        DataType::Int32 => len * 4,
        DataType::Int64 => len * 8,
        DataType::UInt8 => len,
        DataType::UInt16 => len * 2,
        DataType::UInt32 => len * 4,
        DataType::UInt64 => len * 8,
        DataType::Float32 => len * 4,
        DataType::Float64 => len * 8,
        DataType::Date32 => len * 4,
        DataType::Date64 => len * 8,
        DataType::Time32(TimeUnit::Second) => len * 4,
        DataType::Time32(TimeUnit::Millisecond) => len * 4,
        DataType::Time64(TimeUnit::Microsecond) => len * 8,
        DataType::Time64(TimeUnit::Nanosecond) => len * 8,
        // TODO: this is not accurate, but should be good enough for now
        DataType::Utf8 => len * 100 + len * 4,
        DataType::LargeUtf8 => len * 100 + len * 8,
        DataType::Decimal128(_, _) => len * 16,
        DataType::Dictionary(key_type, value_type) => {
            // TODO: this is not accurate, but should be good enough for now
            slot_size(len, key_type.as_ref()) + slot_size(len / 10, value_type.as_ref())
        }
        // TODO: this is not accurate, but should be good enough for now
        DataType::Binary => len * 100 + len * 4,
        DataType::LargeBinary => len * 100 + len * 8,
        DataType::FixedSizeBinary(s) => len * (*s as usize),
        DataType::Timestamp(_, _) => len * 8,
        dt => unimplemented!(
            "{}",
            format!("data type {dt} not supported in shuffle write")
        ),
    }
}

struct SpillInfo {
    file: RefCountedTempFile,
    offsets: Vec<u64>,
}

struct ShuffleRepartitioner {
    output_data_file: String,
    output_index_file: String,
    schema: SchemaRef,
    buffered_partitions: Vec<PartitionBuffer>,
    spills: Mutex<Vec<SpillInfo>>,
    /// Sort expressions
    /// Partitioning scheme to use
    partitioning: Partitioning,
    num_output_partitions: usize,
    runtime: Arc<RuntimeEnv>,
    metrics: ShuffleRepartitionerMetrics,
    reservation: MemoryReservation,
    /// Hashes for each row in the current batch
    hashes_buf: Vec<u32>,
    /// Partition ids for each row in the current batch
    partition_ids: Vec<u64>,
    /// The configured batch size
    batch_size: usize,
}

struct ShuffleRepartitionerMetrics {
    /// metrics
    baseline: BaselineMetrics,

    /// Time to perform repartitioning
    repart_time: Time,

    /// Time interacting with memory pool
    mempool_time: Time,

    /// Time encoding batches to IPC format
    ipc_time: Time,

    /// Time spent writing to disk. Maps to "shuffleWriteTime" in Spark SQL Metrics.
    write_time: Time,

    /// Number of input batches
    input_batches: Count,

    /// count of spills during the execution of the operator
    spill_count: Count,

    /// total spilled bytes during the execution of the operator
    spilled_bytes: Count,

    /// The original size of spilled data. Different to `spilled_bytes` because of compression.
    data_size: Count,
}

impl ShuffleRepartitionerMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            repart_time: MetricBuilder::new(metrics).subset_time("repart_time", partition),
            mempool_time: MetricBuilder::new(metrics).subset_time("mempool_time", partition),
            ipc_time: MetricBuilder::new(metrics).subset_time("ipc_time", partition),
            write_time: MetricBuilder::new(metrics).subset_time("write_time", partition),
            input_batches: MetricBuilder::new(metrics).counter("input_batches", partition),
            spill_count: MetricBuilder::new(metrics).spill_count(partition),
            spilled_bytes: MetricBuilder::new(metrics).spilled_bytes(partition),
            data_size: MetricBuilder::new(metrics).counter("data_size", partition),
        }
    }
}

impl ShuffleRepartitioner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        partition_id: usize,
        output_data_file: String,
        output_index_file: String,
        schema: SchemaRef,
        partitioning: Partitioning,
        metrics: ShuffleRepartitionerMetrics,
        runtime: Arc<RuntimeEnv>,
        batch_size: usize,
    ) -> Self {
        let num_output_partitions = partitioning.partition_count();
        let reservation = MemoryConsumer::new(format!("ShuffleRepartitioner[{}]", partition_id))
            .with_can_spill(true)
            .register(&runtime.memory_pool);

        let mut hashes_buf = Vec::with_capacity(batch_size);
        let mut partition_ids = Vec::with_capacity(batch_size);

        // Safety: `hashes_buf` will be filled with valid values before being used.
        // `partition_ids` will be filled with valid values before being used.
        unsafe {
            hashes_buf.set_len(batch_size);
            partition_ids.set_len(batch_size);
        }

        Self {
            output_data_file,
            output_index_file,
            schema: Arc::clone(&schema),
            buffered_partitions: (0..num_output_partitions)
                .map(|partition_id| {
                    PartitionBuffer::new(Arc::clone(&schema), batch_size, partition_id, &runtime)
                })
                .collect::<Vec<_>>(),
            spills: Mutex::new(vec![]),
            partitioning,
            num_output_partitions,
            runtime,
            metrics,
            reservation,
            hashes_buf,
            partition_ids,
            batch_size,
        }
    }

    /// Shuffles rows in input batch into corresponding partition buffer.
    /// This function will slice input batch according to configured batch size and then
    /// shuffle rows into corresponding partition buffer.
    async fn insert_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let start_time = Instant::now();
        let mut start = 0;
        while start < batch.num_rows() {
            let end = (start + self.batch_size).min(batch.num_rows());
            let batch = batch.slice(start, end - start);
            self.partitioning_batch(batch).await?;
            start = end;
        }
        self.metrics.input_batches.add(1);
        self.metrics
            .baseline
            .elapsed_compute()
            .add_duration(start_time.elapsed());
        Ok(())
    }

    /// Shuffles rows in input batch into corresponding partition buffer.
    /// This function first calculates hashes for rows and then takes rows in same
    /// partition as a record batch which is appended into partition buffer.
    /// This should not be called directly. Use `insert_batch` instead.
    async fn partitioning_batch(&mut self, input: RecordBatch) -> Result<()> {
        if input.num_rows() == 0 {
            // skip empty batch
            return Ok(());
        }

        if input.num_rows() > self.batch_size {
            return Err(DataFusionError::Internal(
                "Input batch size exceeds configured batch size. Call `insert_batch` instead."
                    .to_string(),
            ));
        }

        // Update data size metric
        self.metrics.data_size.add(input.get_array_memory_size());

        // NOTE: in shuffle writer exec, the output_rows metrics represents the
        // number of rows those are written to output data file.
        self.metrics.baseline.record_output(input.num_rows());

        let num_output_partitions = self.num_output_partitions;
        match &self.partitioning {
            Partitioning::Hash(exprs, _) => {
                // evaluate partitioning expressions
                let arrays = exprs
                    .iter()
                    .map(|expr| expr.evaluate(&input)?.into_array(input.num_rows()))
                    .collect::<Result<Vec<_>>>()?;

                // use identical seed as spark hash partition
                let hashes_buf = &mut self.hashes_buf[..arrays[0].len()];
                hashes_buf.fill(42_u32);

                // Hash arrays and compute buckets based on number of partitions
                let partition_ids = &mut self.partition_ids[..arrays[0].len()];
                create_murmur3_hashes(&arrays, hashes_buf)?
                    .iter()
                    .enumerate()
                    .for_each(|(idx, hash)| {
                        partition_ids[idx] = pmod(*hash, num_output_partitions) as u64
                    });

                for i in 0..num_output_partitions {
                    let selection_vector = partition_ids
                        .iter()
                        .map(|idx| *idx == i as u64)
                        .collect_vec();

                    // TODO reserve memory / spilling
                    let selection_vector = BooleanArray::from(selection_vector);
                    let partition_batch = filter_record_batch(&input, &selection_vector)?;
                    if partition_batch.num_rows() > 0 {
                        self.buffered_partitions[i].active.push(partition_batch);
                        if self.buffered_partitions[i]
                            .active
                            .iter()
                            .map(|b| b.num_rows())
                            .sum::<usize>()
                            > self.batch_size
                        {
                            self.buffered_partitions[i].flush(&self.metrics.ipc_time)?;
                        }
                    }
                }
            }
            Partitioning::UnknownPartitioning(n) if *n == 1 => {
                self.buffered_partitions[0].active.push(input);
                if self.buffered_partitions[0]
                    .active
                    .iter()
                    .map(|b| b.num_rows())
                    .sum::<usize>()
                    > self.batch_size
                {
                    self.buffered_partitions[0].flush(&self.metrics.ipc_time)?;
                }
            }
            other => {
                // this should be unreachable as long as the validation logic
                // in the constructor is kept up-to-date
                return Err(DataFusionError::NotImplemented(format!(
                    "Unsupported repartitioning scheme {:?}",
                    other
                )));
            }
        }
        Ok(())
    }

    /// Writes buffered shuffled record batches into Arrow IPC bytes.
    async fn shuffle_write(&mut self) -> Result<SendableRecordBatchStream> {
        let mut elapsed_compute = self.metrics.baseline.elapsed_compute().timer();
        let num_output_partitions = self.num_output_partitions;
        let buffered_partitions = &mut self.buffered_partitions;
        let mut output_batches: Vec<Vec<u8>> = vec![vec![]; num_output_partitions];
        let mut offsets = vec![0; num_output_partitions + 1];
        for i in 0..num_output_partitions {
            buffered_partitions[i].flush(&self.metrics.ipc_time)?;
            output_batches[i] = std::mem::take(&mut buffered_partitions[i].frozen);
        }

        let mut spills = self.spills.lock().await;
        let output_spills = spills.drain(..).collect::<Vec<_>>();

        let data_file = self.output_data_file.clone();
        let index_file = self.output_index_file.clone();

        let mut write_time = self.metrics.write_time.timer();

        let output_data = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(data_file)
            .map_err(|e| DataFusionError::Execution(format!("shuffle write error: {:?}", e)))?;

        let mut output_data = BufWriter::new(output_data);

        for i in 0..num_output_partitions {
            offsets[i] = output_data.stream_position()?;
            output_data.write_all(&output_batches[i])?;
            output_batches[i].clear();

            // append partition in each spills
            for spill in &output_spills {
                let length = spill.offsets[i + 1] - spill.offsets[i];
                if length > 0 {
                    let mut spill_file =
                        BufReader::new(File::open(spill.file.path()).map_err(Self::to_df_err)?);
                    spill_file.seek(SeekFrom::Start(spill.offsets[i]))?;
                    std::io::copy(&mut spill_file.take(length), &mut output_data)
                        .map_err(Self::to_df_err)?;
                }
            }
        }
        output_data.flush()?;

        // add one extra offset at last to ease partition length computation
        offsets[num_output_partitions] = output_data.stream_position().map_err(Self::to_df_err)?;

        let mut output_index =
            BufWriter::new(File::create(index_file).map_err(|e| {
                DataFusionError::Execution(format!("shuffle write error: {:?}", e))
            })?);
        for offset in offsets {
            output_index
                .write_all(&(offset as i64).to_le_bytes()[..])
                .map_err(Self::to_df_err)?;
        }
        output_index.flush()?;

        write_time.stop();

        let mut mempool_timer = self.metrics.mempool_time.timer();
        let used = self.reservation.size();
        self.reservation.shrink(used);
        mempool_timer.stop();

        elapsed_compute.stop();

        // shuffle writer always has empty output
        Ok(Box::pin(EmptyStream::try_new(Arc::clone(&self.schema))?))
    }

    fn to_df_err(e: Error) -> DataFusionError {
        DataFusionError::Execution(format!("shuffle write error: {:?}", e))
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

    async fn spill(&mut self) -> Result<usize> {
        log::debug!(
            "ShuffleRepartitioner spilling shuffle data of {} to disk while inserting ({} time(s) so far)",
            self.used(),
            self.spill_count()
        );

        // we could always get a chance to free some memory as long as we are holding some
        if self.buffered_partitions.is_empty() {
            return Ok(0);
        }

        let mut timer = self.metrics.write_time.timer();

        let spillfile = self
            .runtime
            .disk_manager
            .create_tmp_file("shuffle writer spill")?;
        let offsets = spill_into(
            &mut self.buffered_partitions,
            spillfile.path(),
            self.num_output_partitions,
            &self.metrics.ipc_time,
        )?;

        timer.stop();

        let mut spills = self.spills.lock().await;
        let used = self.reservation.size();
        self.metrics.spill_count.add(1);
        self.metrics.spilled_bytes.add(used);
        spills.push(SpillInfo {
            file: spillfile,
            offsets,
        });
        Ok(used)
    }
}

/// consume the `buffered_partitions` and do spill into a single temp shuffle output file
fn spill_into(
    buffered_partitions: &mut [PartitionBuffer],
    path: &Path,
    num_output_partitions: usize,
    ipc_time: &Time,
) -> Result<Vec<u64>> {
    let mut output_batches: Vec<Vec<u8>> = vec![vec![]; num_output_partitions];

    for i in 0..num_output_partitions {
        buffered_partitions[i].flush(ipc_time)?;
        output_batches[i] = std::mem::take(&mut buffered_partitions[i].frozen);
    }
    let path = path.to_owned();

    let mut offsets = vec![0; num_output_partitions + 1];
    let mut spill_data = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| DataFusionError::Execution(format!("Error occurred while spilling {}", e)))?;

    for i in 0..num_output_partitions {
        offsets[i] = spill_data.stream_position()?;
        spill_data.write_all(&output_batches[i])?;
        output_batches[i].clear();
    }
    // add one extra offset at last to ease partition length computation
    offsets[num_output_partitions] = spill_data.stream_position()?;
    Ok(offsets)
}

impl Debug for ShuffleRepartitioner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShuffleRepartitioner")
            .field("memory_used", &self.used())
            .field("spilled_bytes", &self.spilled_bytes())
            .field("spilled_count", &self.spill_count())
            .field("data_size", &self.data_size())
            .finish()
    }
}

async fn external_shuffle(
    mut input: SendableRecordBatchStream,
    partition_id: usize,
    output_data_file: String,
    output_index_file: String,
    partitioning: Partitioning,
    metrics: ShuffleRepartitionerMetrics,
    context: Arc<TaskContext>,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let mut repartitioner = ShuffleRepartitioner::new(
        partition_id,
        output_data_file,
        output_index_file,
        Arc::clone(&schema),
        partitioning,
        metrics,
        context.runtime_env(),
        context.session_config().batch_size(),
    );

    while let Some(batch) = input.next().await {
        // Block on the repartitioner to insert the batch and shuffle the rows
        // into the corresponding partition buffer.
        // Otherwise, pull the next batch from the input stream might overwrite the
        // current batch in the repartitioner.
        block_on(repartitioner.insert_batch(batch?))?;
    }
    repartitioner.shuffle_write().await
}

/// Checksum algorithms for writing IPC bytes.
#[derive(Clone)]
pub(crate) enum Checksum {
    /// CRC32 checksum algorithm.
    CRC32(Hasher),
    /// Adler32 checksum algorithm.
    Adler32(Adler32),
}

impl Checksum {
    pub(crate) fn try_new(algo: i32, initial_opt: Option<u32>) -> CometResult<Self> {
        match algo {
            0 => {
                let hasher = if let Some(initial) = initial_opt {
                    Hasher::new_with_initial(initial)
                } else {
                    Hasher::new()
                };
                Ok(Checksum::CRC32(hasher))
            }
            1 => {
                let hasher = if let Some(initial) = initial_opt {
                    // Note that Adler32 initial state is not zero.
                    // i.e., `Adler32::from_checksum(0)` is not the same as `Adler32::new()`.
                    Adler32::from_checksum(initial)
                } else {
                    Adler32::new()
                };
                Ok(Checksum::Adler32(hasher))
            }
            _ => Err(CometError::Internal(
                "Unsupported checksum algorithm".to_string(),
            )),
        }
    }

    pub(crate) fn update(&mut self, cursor: &mut Cursor<&mut Vec<u8>>) -> CometResult<()> {
        match self {
            Checksum::CRC32(hasher) => {
                std::io::Seek::seek(cursor, SeekFrom::Start(0))?;
                hasher.update(cursor.chunk());
                Ok(())
            }
            Checksum::Adler32(hasher) => {
                std::io::Seek::seek(cursor, SeekFrom::Start(0))?;
                hasher.write(cursor.chunk());
                Ok(())
            }
        }
    }

    pub(crate) fn finalize(self) -> u32 {
        match self {
            Checksum::CRC32(hasher) => hasher.finalize(),
            Checksum::Adler32(hasher) => hasher.finish(),
        }
    }
}

/// Writes given record batch as Arrow IPC bytes into given writer.
/// Returns number of bytes written.
pub(crate) fn write_ipc_compressed<W: Write + Seek>(
    batch: &RecordBatch,
    output: &mut W,
    ipc_time: &Time,
) -> Result<usize> {
    if batch.num_rows() == 0 {
        return Ok(0);
    }

    let mut timer = ipc_time.timer();
    let start_pos = output.stream_position()?;

    // write ipc_length placeholder
    output.write_all(&[0u8; 8])?;

    // write ipc data
    // TODO: make compression level configurable
    let mut arrow_writer = StreamWriter::try_new(zstd::Encoder::new(output, 1)?, &batch.schema())?;
    arrow_writer.write(batch)?;
    arrow_writer.finish()?;

    let zwriter = arrow_writer.into_inner()?;
    let output = zwriter.finish()?;
    let end_pos = output.stream_position()?;
    let ipc_length = end_pos - start_pos - 8;

    // fill ipc length
    output.seek(SeekFrom::Start(start_pos))?;
    output.write_all(&ipc_length.to_le_bytes()[..])?;
    output.seek(SeekFrom::Start(end_pos))?;

    timer.stop();

    Ok((end_pos - start_pos) as usize)
}

/// A stream that yields no record batches which represent end of output.
pub struct EmptyStream {
    /// Schema representing the data
    schema: SchemaRef,
}

impl EmptyStream {
    /// Create an iterator for a vector of record batches
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        Ok(Self { schema })
    }
}

impl Stream for EmptyStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

impl RecordBatchStream for EmptyStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

fn pmod(hash: u32, n: usize) -> usize {
    let hash = hash as i32;
    let n = n as i32;
    let r = hash % n;
    let result = if r < 0 { (r + n) % n } else { r };
    result as usize
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::physical_plan::common::collect;
    use datafusion::physical_plan::memory::MemoryExec;
    use datafusion::prelude::SessionContext;
    use datafusion_execution::config::SessionConfig;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::expressions::Column;
    use tokio::runtime::Runtime;

    #[test]
    fn test_slot_size() {
        let batch_size = 1usize;
        // not inclusive of all supported types, but enough to test the function
        let supported_primitive_types = [
            DataType::Int32,
            DataType::Int64,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Boolean,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::FixedSizeBinary(16),
        ];
        let expected_slot_size = [4, 8, 4, 8, 4, 8, 1, 104, 108, 104, 108, 16];
        supported_primitive_types
            .iter()
            .zip(expected_slot_size.iter())
            .for_each(|(data_type, expected)| {
                let slot_size = slot_size(batch_size, data_type);
                assert_eq!(slot_size, *expected);
            })
    }

    #[test]
    #[cfg_attr(miri, ignore)] // miri can't call foreign function `ZSTD_createCCtx`
    fn test_insert_larger_batch() {
        shuffle_write_test(10000, 1, 16, None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // miri can't call foreign function `ZSTD_createCCtx`
    fn test_insert_smaller_batch() {
        shuffle_write_test(1000, 1, 16, None);
        shuffle_write_test(1000, 10, 16, None);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // miri can't call foreign function `ZSTD_createCCtx`
    #[cfg(not(target_os = "macos"))] // Github MacOS runner fails with "Too many open files".
    fn test_large_number_of_partitions() {
        shuffle_write_test(10000, 10, 200, Some(10 * 1024 * 1024));
        shuffle_write_test(10000, 10, 2000, Some(10 * 1024 * 1024));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // miri can't call foreign function `ZSTD_createCCtx`
    #[cfg(not(target_os = "macos"))] // Github MacOS runner fails with "Too many open files".
    fn test_large_number_of_partitions_spilling() {
        shuffle_write_test(10000, 100, 200, Some(10 * 1024 * 1024));
    }

    fn shuffle_write_test(
        batch_size: usize,
        num_batches: usize,
        num_partitions: usize,
        memory_limit: Option<usize>,
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
        let mut b = StringBuilder::new();
        for i in 0..batch_size {
            b.append_value(format!("{i}"));
        }
        let array = b.finish();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();

        let batches = (0..num_batches).map(|_| batch.clone()).collect::<Vec<_>>();

        let partitions = &[batches];
        let exec = ShuffleWriterExec::try_new(
            Arc::new(MemoryExec::try_new(partitions, batch.schema(), None).unwrap()),
            Partitioning::Hash(vec![Arc::new(Column::new("a", 0))], num_partitions),
            "/tmp/data.out".to_string(),
            "/tmp/index.out".to_string(),
        )
        .unwrap();

        // 10MB memory should be enough for running this test
        let config = SessionConfig::new();
        let mut runtime_env_builder = RuntimeEnvBuilder::new();
        runtime_env_builder = match memory_limit {
            Some(limit) => runtime_env_builder.with_memory_limit(limit, 1.0),
            None => runtime_env_builder,
        };
        let runtime_env = Arc::new(runtime_env_builder.build().unwrap());
        let ctx = SessionContext::new_with_config_rt(config, runtime_env);
        let task_ctx = ctx.task_ctx();
        let stream = exec.execute(0, task_ctx).unwrap();
        let rt = Runtime::new().unwrap();
        rt.block_on(collect(stream)).unwrap();
    }

    #[test]
    fn test_pmod() {
        let i: Vec<u32> = vec![0x99f0149d, 0x9c67b85d, 0xc8008529, 0xa05b5d7b, 0xcd1e64fb];
        let result = i.into_iter().map(|i| pmod(i, 200)).collect::<Vec<usize>>();

        // expected partition from Spark with n=200
        let expected = vec![69, 5, 193, 171, 115];
        assert_eq!(result, expected);
    }
}
