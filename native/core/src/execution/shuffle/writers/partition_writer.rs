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
use arrow::ipc::writer::StreamWriter;
use datafusion::common::DataFusionError;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use std::fs::{File, OpenOptions};

pub(crate) struct PartitionWriter {
    schema: SchemaRef,
    temp_file: Option<RefCountedTempFile>,
    /// Arrow IPC writer for uncompressed spill data
    ipc_writer: Option<StreamWriter<File>>,
    /// Coalescer for small batches (avoids many tiny IPC messages)
    coalescer: Option<BatchCoalescer>,
    batch_size: usize,
}

impl PartitionWriter {
    pub(crate) fn try_new(schema: SchemaRef, batch_size: usize) -> datafusion::common::Result<Self> {
        Ok(Self {
            schema,
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
    ) -> datafusion::common::Result<&mut StreamWriter<File>> {
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
            self.ipc_writer = Some(StreamWriter::try_new(file, &self.schema)?);
        }
        Ok(self.ipc_writer.as_mut().unwrap())
    }

    /// Write a batch to this partition's spill file, lazily creating the file and writer.
    /// Large batches (>= batch_size rows) bypass the coalescer for zero-copy writes.
    pub(crate) fn write_batch(
        &mut self,
        batch: &RecordBatch,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<usize> {
        let mem_size = batch.get_array_memory_size();

        if batch.num_rows() >= self.batch_size {
            // Large batch fast path: flush coalescer, then write directly (no clone)
            self.flush_coalescer(runtime, metrics)?;
            let writer = self.ensure_writer(runtime)?;
            let mut write_timer = metrics.write_time.timer();
            writer.write(batch)?;
            write_timer.stop();
        } else {
            // Small batch path: push to coalescer, drain completed batches
            let coalescer = self
                .coalescer
                .get_or_insert_with(|| BatchCoalescer::new(batch.schema(), self.batch_size));
            coalescer.push_batch(batch.clone())?;

            let mut completed = Vec::new();
            while let Some(batch) = coalescer.next_completed_batch() {
                completed.push(batch);
            }

            if !completed.is_empty() {
                let writer = self.ensure_writer(runtime)?;
                let mut write_timer = metrics.write_time.timer();
                for batch in &completed {
                    writer.write(batch)?;
                }
                write_timer.stop();
            }
        }

        Ok(mem_size)
    }

    /// Flush remaining rows in the coalescer to the IPC writer.
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
            let mut write_timer = metrics.write_time.timer();
            for batch in &remaining {
                writer.write(batch)?;
            }
            write_timer.stop();
        }
        Ok(())
    }

    /// Flush the coalescer to the IPC writer (used during spill).
    pub(crate) fn flush(
        &mut self,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        self.flush_coalescer(runtime, metrics)
    }

    /// Flush the coalescer and finalize the IPC stream (writes EOS marker).
    /// Must be called before reading back from the spill file.
    pub(crate) fn finish(
        &mut self,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        self.flush_coalescer(runtime, metrics)?;
        if let Some(mut writer) = self.ipc_writer.take() {
            let mut write_timer = metrics.write_time.timer();
            writer.finish()?;
            write_timer.stop();
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
