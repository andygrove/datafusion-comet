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

//! Infrastructure for the Grace Hash Join operator.
//!
//! Provides spill-to-disk support (write/read Arrow IPC), efficient
//! hash-partitioning via a prefix-sum algorithm, and helper ExecutionPlan
//! wrappers for feeding data into DataFusion's HashJoinExec.

use std::any::Any;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use std::sync::Mutex;

use ahash::RandomState;
use arrow::array::UInt32Array;
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use datafusion::common::hash_utils::create_hashes;
use datafusion::common::{DataFusionError, Result as DFResult};
use datafusion::execution::context::TaskContext;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::stream;
use tokio::sync::mpsc;

/// I/O buffer size for spill file reads and writes (1 MB).
pub(crate) const SPILL_IO_BUFFER_SIZE: usize = 1024 * 1024;

/// Target number of rows per coalesced batch when reading spill files.
pub(crate) const SPILL_READ_COALESCE_TARGET: usize = 8192;

/// Random state for hashing join keys into partitions. Uses fixed seeds
/// different from DataFusion's HashJoinExec to avoid correlation.
/// The `recursion_level` is XORed into the seed so that recursive
/// repartitioning uses different hash functions at each level.
pub(crate) fn partition_random_state(recursion_level: usize) -> RandomState {
    RandomState::with_seeds(
        0x517cc1b727220a95 ^ (recursion_level as u64),
        0x3a8b7c9d1e2f4056,
        0,
        0,
    )
}

// ---------------------------------------------------------------------------
// SpillWriter: incremental append to Arrow IPC spill files
// ---------------------------------------------------------------------------

/// Wraps an Arrow IPC `StreamWriter` for incremental spill writes with
/// LZ4 compression. Avoids the O(n²) read-rewrite pattern by keeping
/// the writer open for appends.
pub(crate) struct SpillWriter {
    writer: StreamWriter<BufWriter<File>>,
    temp_file: RefCountedTempFile,
    bytes_written: usize,
}

impl SpillWriter {
    /// Create a new spill writer backed by a temp file.
    pub fn new(temp_file: RefCountedTempFile, schema: &SchemaRef) -> DFResult<Self> {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(temp_file.path())
            .map_err(|e| DataFusionError::Execution(format!("Failed to open spill file: {e}")))?;
        let buf_writer = BufWriter::with_capacity(SPILL_IO_BUFFER_SIZE, file);
        let write_options =
            IpcWriteOptions::default().try_with_compression(Some(CompressionType::LZ4_FRAME))?;
        let writer = StreamWriter::try_new_with_options(buf_writer, schema, write_options)?;
        Ok(Self {
            writer,
            temp_file,
            bytes_written: 0,
        })
    }

    /// Append a single batch to the spill file.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> DFResult<()> {
        if batch.num_rows() > 0 {
            self.bytes_written += batch.get_array_memory_size();
            self.writer.write(batch)?;
        }
        Ok(())
    }

    /// Append multiple batches to the spill file.
    pub fn write_batches(&mut self, batches: &[RecordBatch]) -> DFResult<()> {
        for batch in batches {
            self.write_batch(batch)?;
        }
        Ok(())
    }

    /// Finish writing. Must be called before reading back.
    /// Returns the temp file handle and total bytes written.
    pub fn finish(mut self) -> DFResult<(RefCountedTempFile, usize)> {
        self.writer.finish()?;
        Ok((self.temp_file, self.bytes_written))
    }
}

// ---------------------------------------------------------------------------
// SpillReaderExec: streaming ExecutionPlan for reading spill files
// ---------------------------------------------------------------------------

/// An ExecutionPlan that streams record batches from an Arrow IPC spill file.
/// Reads on a blocking thread via `spawn_blocking` to avoid blocking the
/// async executor. Coalesces small sub-batches into ~8192-row chunks.
#[derive(Debug)]
pub(crate) struct SpillReaderExec {
    spill_file: RefCountedTempFile,
    schema: SchemaRef,
    cache: PlanProperties,
}

impl SpillReaderExec {
    pub fn new(spill_file: RefCountedTempFile, schema: SchemaRef) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            spill_file,
            schema,
            cache,
        }
    }
}

impl DisplayAs for SpillReaderExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SpillReaderExec")
    }
}

impl ExecutionPlan for SpillReaderExec {
    fn name(&self) -> &str {
        "SpillReaderExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let schema = Arc::clone(&self.schema);
        let coalesce_schema = Arc::clone(&self.schema);
        let path = self.spill_file.path().to_path_buf();
        // Keep the temp file alive until the reader is done.
        let spill_file_handle = self.spill_file.clone();

        let (tx, rx) = mpsc::channel::<DFResult<RecordBatch>>(4);

        tokio::task::spawn_blocking(move || {
            let _keep_alive = spill_file_handle;
            let file = match File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.blocking_send(Err(DataFusionError::Execution(format!(
                        "Failed to open spill file: {e}"
                    ))));
                    return;
                }
            };
            let reader = match StreamReader::try_new(
                BufReader::with_capacity(SPILL_IO_BUFFER_SIZE, file),
                None,
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(Err(DataFusionError::ArrowError(Box::new(e), None)));
                    return;
                }
            };

            // Coalesce small sub-batches into larger ones.
            let mut pending: Vec<RecordBatch> = Vec::new();
            let mut pending_rows = 0usize;

            for batch_result in reader {
                let batch = match batch_result {
                    Ok(b) => b,
                    Err(e) => {
                        let _ =
                            tx.blocking_send(Err(DataFusionError::ArrowError(Box::new(e), None)));
                        return;
                    }
                };
                if batch.num_rows() == 0 {
                    continue;
                }
                pending_rows += batch.num_rows();
                pending.push(batch);

                if pending_rows >= SPILL_READ_COALESCE_TARGET {
                    let merged = if pending.len() == 1 {
                        Ok(pending.pop().unwrap())
                    } else {
                        concat_batches(&coalesce_schema, &pending)
                            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                    };
                    pending.clear();
                    pending_rows = 0;
                    if tx.blocking_send(merged).is_err() {
                        return;
                    }
                }
            }

            // Flush remaining
            if !pending.is_empty() {
                let merged = if pending.len() == 1 {
                    Ok(pending.pop().unwrap())
                } else {
                    concat_batches(&coalesce_schema, &pending)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
                };
                let _ = tx.blocking_send(merged);
            }
        });

        let batch_stream = futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|batch| (batch, rx))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            batch_stream,
        )))
    }
}

// ---------------------------------------------------------------------------
// StreamSourceExec: wrap an existing stream as an ExecutionPlan
// ---------------------------------------------------------------------------

/// An ExecutionPlan that yields batches from a pre-existing stream.
/// Unlike `DataSourceExec(MemorySourceConfig)`, this does NOT wrap its
/// output in `BatchSplitStream`, avoiding Arrow i32 offset overflow
/// and memory over-counting from zero-copy slices.
pub(crate) struct StreamSourceExec {
    stream: Mutex<Option<SendableRecordBatchStream>>,
    schema: SchemaRef,
    cache: PlanProperties,
}

impl StreamSourceExec {
    pub fn new(stream: SendableRecordBatchStream, schema: SchemaRef) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            datafusion::physical_plan::execution_plan::EmissionType::Incremental,
            datafusion::physical_plan::execution_plan::Boundedness::Bounded,
        );
        Self {
            stream: Mutex::new(Some(stream)),
            schema,
            cache,
        }
    }
}

impl fmt::Debug for StreamSourceExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StreamSourceExec").finish()
    }
}

impl DisplayAs for StreamSourceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StreamSourceExec")
    }
}

impl ExecutionPlan for StreamSourceExec {
    fn name(&self) -> &str {
        "StreamSourceExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        self.stream
            .lock()
            .map_err(|e| DataFusionError::Internal(format!("lock poisoned: {e}")))?
            .take()
            .ok_or_else(|| {
                DataFusionError::Internal("StreamSourceExec: stream already consumed".to_string())
            })
    }
}

// ---------------------------------------------------------------------------
// memory_source_exec helper
// ---------------------------------------------------------------------------

/// Create a `StreamSourceExec` that yields `data` batches without splitting.
pub(crate) fn memory_source_exec(
    data: Vec<RecordBatch>,
    schema: &SchemaRef,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let schema_clone = Arc::clone(schema);
    let stream =
        RecordBatchStreamAdapter::new(Arc::clone(schema), stream::iter(data.into_iter().map(Ok)));
    Ok(Arc::new(StreamSourceExec::new(
        Box::pin(stream),
        schema_clone,
    )))
}

// ---------------------------------------------------------------------------
// read_spilled_batches helper
// ---------------------------------------------------------------------------

/// Read all record batches from a finished spill file eagerly.
pub(crate) fn read_spilled_batches(spill_file: &RefCountedTempFile) -> DFResult<Vec<RecordBatch>> {
    let file = File::open(spill_file.path())
        .map_err(|e| DataFusionError::Execution(format!("Failed to open spill file: {e}")))?;
    let reader = BufReader::with_capacity(SPILL_IO_BUFFER_SIZE, file);
    let stream_reader = StreamReader::try_new(reader, None)?;
    let batches: Vec<RecordBatch> = stream_reader.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(batches)
}

// ---------------------------------------------------------------------------
// ScratchSpace: reusable buffers for efficient hash partitioning
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for partitioning batches. Uses a prefix-sum
/// algorithm to compute contiguous row-index regions per partition in a
/// single pass, avoiding N separate `take()` kernel calls.
#[derive(Default)]
pub(crate) struct ScratchSpace {
    hashes: Vec<u64>,
    partition_ids: Vec<u32>,
    partition_row_indices: Vec<u32>,
    /// `partition_starts[k]..partition_starts[k+1]` gives the slice of
    /// `partition_row_indices` belonging to partition k.
    partition_starts: Vec<u32>,
}

impl ScratchSpace {
    /// Compute hashes and partition ids, then build the prefix-sum index
    /// structures for the given batch.
    pub fn compute_partitions(
        &mut self,
        batch: &RecordBatch,
        keys: &[Arc<dyn PhysicalExpr>],
        num_partitions: usize,
        recursion_level: usize,
    ) -> DFResult<()> {
        let num_rows = batch.num_rows();

        let key_columns: Vec<_> = keys
            .iter()
            .map(|expr| expr.evaluate(batch).and_then(|cv| cv.into_array(num_rows)))
            .collect::<DFResult<Vec<_>>>()?;

        self.hashes.resize(num_rows, 0);
        self.hashes.fill(0);
        let random_state = partition_random_state(recursion_level);
        create_hashes(&key_columns, &random_state, &mut self.hashes)?;

        self.partition_ids.resize(num_rows, 0);
        for (i, hash) in self.hashes[..num_rows].iter().enumerate() {
            self.partition_ids[i] = (*hash as u32) % (num_partitions as u32);
        }

        self.map_partition_ids_to_starts_and_indices(num_partitions, num_rows);

        Ok(())
    }

    /// Prefix-sum algorithm: count → accumulate → scatter (reverse).
    fn map_partition_ids_to_starts_and_indices(&mut self, num_partitions: usize, num_rows: usize) {
        let partition_ids = &self.partition_ids[..num_rows];

        let partition_counters = &mut self.partition_starts;
        partition_counters.resize(num_partitions + 1, 0);
        partition_counters.fill(0);
        partition_ids
            .iter()
            .for_each(|pid| partition_counters[*pid as usize] += 1);

        let mut accum = 0u32;
        for v in partition_counters.iter_mut() {
            *v += accum;
            accum = *v;
        }

        self.partition_row_indices.resize(num_rows, 0);
        for (index, pid) in partition_ids.iter().enumerate().rev() {
            self.partition_starts[*pid as usize] -= 1;
            let pos = self.partition_starts[*pid as usize];
            self.partition_row_indices[pos as usize] = index as u32;
        }
    }

    /// Get the row index slice for a given partition.
    pub fn partition_slice(&self, partition_id: usize) -> &[u32] {
        let start = self.partition_starts[partition_id] as usize;
        let end = self.partition_starts[partition_id + 1] as usize;
        &self.partition_row_indices[start..end]
    }

    /// Number of rows in a given partition.
    pub fn partition_len(&self, partition_id: usize) -> usize {
        (self.partition_starts[partition_id + 1] - self.partition_starts[partition_id]) as usize
    }

    /// Extract rows for a partition from a batch using `take()`.
    pub fn take_partition(
        &self,
        batch: &RecordBatch,
        partition_id: usize,
    ) -> DFResult<Option<RecordBatch>> {
        let row_indices = self.partition_slice(partition_id);
        if row_indices.is_empty() {
            return Ok(None);
        }
        let indices_array = UInt32Array::from(row_indices.to_vec());
        let columns: Vec<_> = batch
            .columns()
            .iter()
            .map(|col| take(col.as_ref(), &indices_array, None))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(RecordBatch::try_new(batch.schema(), columns)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::physical_expr::expressions::Column;
    use datafusion::prelude::SessionContext;
    use futures::TryStreamExt;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("val", DataType::Utf8, false),
        ]))
    }

    fn make_batch(ids: &[i32], values: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            test_schema(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_spill_write_and_read() {
        let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
        let temp_file = runtime.disk_manager.create_tmp_file("test").unwrap();
        let schema = test_schema();

        let batch1 = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        let batch2 = make_batch(&[4, 5], &["d", "e"]);

        let mut writer = SpillWriter::new(temp_file, &schema).unwrap();
        writer.write_batch(&batch1).unwrap();
        writer.write_batch(&batch2).unwrap();
        let (file, bytes_written) = writer.finish().unwrap();

        assert!(bytes_written > 0);

        let batches = read_spilled_batches(&file).unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 5);
    }

    #[test]
    fn test_spill_empty_batches_skipped() {
        let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
        let temp_file = runtime.disk_manager.create_tmp_file("test").unwrap();
        let schema = test_schema();

        let batch = make_batch(&[1], &["a"]);
        let empty = RecordBatch::new_empty(Arc::clone(&schema));

        let mut writer = SpillWriter::new(temp_file, &schema).unwrap();
        writer.write_batch(&empty).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.write_batch(&empty).unwrap();
        let (file, _) = writer.finish().unwrap();

        let batches = read_spilled_batches(&file).unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_spill_reader_exec_streams_batches() {
        let runtime = RuntimeEnvBuilder::new().build_arc().unwrap();
        let temp_file = runtime.disk_manager.create_tmp_file("test").unwrap();
        let schema = test_schema();

        // Write many small batches to test coalescing
        let mut writer = SpillWriter::new(temp_file, &schema).unwrap();
        for i in 0..100 {
            writer.write_batch(&make_batch(&[i], &["x"])).unwrap();
        }
        let (file, _) = writer.finish().unwrap();

        let reader = SpillReaderExec::new(file, Arc::clone(&schema));
        let ctx = SessionContext::new();
        let stream = reader.execute(0, ctx.task_ctx()).unwrap();
        let result: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
        // Coalescing should produce fewer than 100 batches
        assert!(result.len() < 100);
    }

    #[tokio::test]
    async fn test_stream_source_exec() {
        let schema = test_schema();
        let batch = make_batch(&[1, 2, 3], &["a", "b", "c"]);
        let input_stream = RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            stream::iter(vec![Ok(batch.clone())]),
        );

        let exec = StreamSourceExec::new(Box::pin(input_stream), Arc::clone(&schema));
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 3);
    }

    #[tokio::test]
    async fn test_stream_source_exec_consumed_twice_errors() {
        let schema = test_schema();
        let input_stream = RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            stream::iter(vec![Ok(make_batch(&[1], &["a"]))]),
        );

        let exec = Arc::new(StreamSourceExec::new(
            Box::pin(input_stream),
            Arc::clone(&schema),
        ));
        let ctx = SessionContext::new();

        // First call succeeds
        let _stream = exec.execute(0, ctx.task_ctx()).unwrap();
        // Second call should error
        let result = exec.execute(0, ctx.task_ctx());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_source_exec() {
        let schema = test_schema();
        let batches = vec![make_batch(&[1, 2], &["a", "b"]), make_batch(&[3], &["c"])];

        let exec = memory_source_exec(batches, &schema).unwrap();
        let ctx = SessionContext::new();
        let stream = exec.execute(0, ctx.task_ctx()).unwrap();
        let result: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);
    }

    #[test]
    fn test_scratch_space_partitioning() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7]))],
        )
        .unwrap();

        let key = Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let mut scratch = ScratchSpace::default();
        scratch.compute_partitions(&batch, &[key], 4, 0).unwrap();

        // All 8 rows should be distributed across 4 partitions
        let mut total = 0;
        for p in 0..4 {
            total += scratch.partition_len(p);
        }
        assert_eq!(total, 8);

        // Each partition should produce valid sub-batches
        for p in 0..4 {
            if scratch.partition_len(p) > 0 {
                let sub = scratch.take_partition(&batch, p).unwrap().unwrap();
                assert_eq!(sub.num_rows(), scratch.partition_len(p));
                assert_eq!(sub.num_columns(), 1);
            }
        }
    }

    #[test]
    fn test_scratch_space_all_same_partition() {
        // When all rows hash to the same partition, one partition gets
        // everything and the rest get nothing.
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        // Use the same value so all rows hash identically
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![42, 42, 42, 42]))],
        )
        .unwrap();

        let key = Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let mut scratch = ScratchSpace::default();
        scratch.compute_partitions(&batch, &[key], 4, 0).unwrap();

        let mut nonempty = 0;
        for p in 0..4 {
            if scratch.partition_len(p) > 0 {
                assert_eq!(scratch.partition_len(p), 4);
                nonempty += 1;
            }
        }
        assert_eq!(nonempty, 1);
    }

    #[test]
    fn test_different_recursion_levels_produce_different_partitions() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]))],
        )
        .unwrap();

        let key = Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>;
        let mut scratch = ScratchSpace::default();

        scratch
            .compute_partitions(&batch, std::slice::from_ref(&key), 4, 0)
            .unwrap();
        let level0: Vec<usize> = (0..4).map(|p| scratch.partition_len(p)).collect();

        scratch
            .compute_partitions(&batch, std::slice::from_ref(&key), 4, 1)
            .unwrap();
        let level1: Vec<usize> = (0..4).map(|p| scratch.partition_len(p)).collect();

        // Different seeds should generally produce different distributions
        // (not guaranteed for small inputs, but very likely for 8 distinct values)
        assert_ne!(
            level0, level1,
            "Different recursion levels should use different hash seeds"
        );
    }
}
