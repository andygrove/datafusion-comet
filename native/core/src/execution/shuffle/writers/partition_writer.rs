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

use crate::execution::shuffle::metrics::ShufflePartitionerMetrics;
use arrow::array::RecordBatch;
use arrow::compute::kernels::coalesce::BatchCoalescer;
use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use datafusion::common::DataFusionError;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use std::fs::{File, OpenOptions};
use std::io::BufWriter;

/// Writes batches for a single partition to a spill file as a single Arrow IPC stream.
/// Small batches are coalesced via `BatchCoalescer` before writing.
pub(crate) struct PartitionWriter {
    schema: SchemaRef,
    ipc_options: IpcWriteOptions,
    temp_file: Option<RefCountedTempFile>,
    ipc_writer: Option<StreamWriter<BufWriter<File>>>,
    coalescer: Option<BatchCoalescer>,
    batch_size: usize,
}

impl PartitionWriter {
    pub(crate) fn try_new(
        schema: SchemaRef,
        batch_size: usize,
        ipc_options: IpcWriteOptions,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            schema,
            ipc_options,
            temp_file: None,
            ipc_writer: None,
            coalescer: None,
            batch_size,
        })
    }

    /// Ensure the spill file and IPC writer exist, creating them lazily.
    fn ensure_writer(
        &mut self,
        runtime: &RuntimeEnv,
    ) -> datafusion::common::Result<&mut StreamWriter<BufWriter<File>>> {
        if self.ipc_writer.is_none() {
            let temp_file = runtime
                .disk_manager
                .create_tmp_file("shuffle writer spill")?;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(temp_file.path())
                .map_err(|e| {
                    DataFusionError::Execution(format!("Error occurred while spilling {e}"))
                })?;
            self.temp_file = Some(temp_file);
            let writer = StreamWriter::try_new_with_options(
                BufWriter::new(file),
                &self.schema,
                self.ipc_options.clone(),
            )?;
            self.ipc_writer = Some(writer);
        }
        Ok(self.ipc_writer.as_mut().unwrap())
    }

    /// Write a batch to this partition's spill file.
    /// Large batches (>= batch_size) are written directly; small batches are coalesced first.
    /// Returns the batch's array memory size for memory accounting.
    pub(crate) fn write_batch(
        &mut self,
        batch: &RecordBatch,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }

        let mem_size = batch.get_array_memory_size();

        if batch.num_rows() >= self.batch_size {
            // Large batch: flush coalescer first, then write directly (no clone)
            self.flush_coalescer(runtime, metrics)?;
            let writer = self.ensure_writer(runtime)?;
            let mut timer = metrics.encode_time.timer();
            writer.write(batch)?;
            timer.stop();
        } else {
            // Small batch: push to coalescer
            let coalescer = self
                .coalescer
                .get_or_insert_with(|| BatchCoalescer::new(batch.schema(), self.batch_size));
            coalescer.push_batch(batch.clone())?;

            // Drain completed batches to the IPC writer
            let mut completed = Vec::new();
            while let Some(batch) = coalescer.next_completed_batch() {
                completed.push(batch);
            }
            if !completed.is_empty() {
                let writer = self.ensure_writer(runtime)?;
                let mut timer = metrics.encode_time.timer();
                for batch in &completed {
                    writer.write(batch)?;
                }
                timer.stop();
            }
        }

        Ok(mem_size)
    }

    /// Flush the coalescer's remaining buffered rows to the IPC writer (for spill).
    fn flush_coalescer(
        &mut self,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        let mut remaining = Vec::new();
        if let Some(coalescer) = &mut self.coalescer {
            coalescer.finish_buffered_batch()?;
            while let Some(batch) = coalescer.next_completed_batch() {
                remaining.push(batch);
            }
        }
        if !remaining.is_empty() {
            let writer = self.ensure_writer(runtime)?;
            let mut timer = metrics.encode_time.timer();
            for batch in &remaining {
                writer.write(batch)?;
            }
            timer.stop();
        }
        Ok(())
    }

    /// Flush the coalescer (for spill). Does NOT finish the IPC stream,
    /// so more batches can be written after spilling.
    pub(crate) fn flush(
        &mut self,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        self.flush_coalescer(runtime, metrics)
    }

    /// Flush coalescer and finish the IPC stream (writes EOS marker).
    /// After this, no more batches can be written.
    pub(crate) fn finish(
        &mut self,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        self.flush_coalescer(runtime, metrics)?;
        if let Some(mut writer) = self.ipc_writer.take() {
            let mut timer = metrics.encode_time.timer();
            writer.finish()?;
            timer.stop();
        }
        Ok(())
    }

    pub(crate) fn path(&self) -> Option<&std::path::Path> {
        self.temp_file.as_ref().map(|t| t.path())
    }

    #[cfg(test)]
    pub(crate) fn has_spill_file(&self) -> bool {
        self.temp_file.is_some()
    }
}
