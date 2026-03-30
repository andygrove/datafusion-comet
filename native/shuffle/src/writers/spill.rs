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

use super::ShuffleBlockWriter;
use crate::writers::buf_batch_writer::BufBatchWriter;
use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::physical_plan::metrics::Time;
use std::fs::{File, OpenOptions};
use std::io::Write;

/// A temporary disk file for spilling a partition's intermediate shuffle data.
struct SpillFile {
    temp_file: RefCountedTempFile,
    file: File,
}

/// Manages encoding, in-memory buffering, and optional disk spilling for a single shuffle
/// partition. Batches are serialized to compressed IPC blocks in an in-memory buffer. When
/// memory pressure triggers a spill, the buffer is written to a temp file on disk.
pub(crate) struct PartitionWriter {
    /// Accumulates compressed IPC blocks in memory.
    buf_writer: BufBatchWriter<ShuffleBlockWriter, std::io::Sink>,
    /// Spill file for intermediate shuffle output for this partition. Created lazily
    /// on first spill. Each spill event appends to this file.
    spill_file: Option<SpillFile>,
}

impl PartitionWriter {
    pub(crate) fn new(shuffle_block_writer: ShuffleBlockWriter, batch_size: usize) -> Self {
        Self {
            buf_writer: BufBatchWriter::new(
                shuffle_block_writer,
                std::io::sink(),
                usize::MAX,
                batch_size,
            ),
            spill_file: None,
        }
    }

    /// Write a batch to the in-memory buffer (serialized as compressed IPC).
    pub(crate) fn write(
        &mut self,
        batch: &RecordBatch,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<usize> {
        self.buf_writer
            .write(batch, encode_time, write_time, coalesce_time)
    }

    /// Returns total estimated memory: serialized IPC buffer + unencoded coalescer batches.
    pub(crate) fn buffered_bytes(&self) -> usize {
        self.buf_writer.buffer_len() + self.buf_writer.coalescer_buffered_bytes()
    }

    fn ensure_spill_file_created(
        &mut self,
        runtime: &RuntimeEnv,
    ) -> datafusion::common::Result<()> {
        if self.spill_file.is_none() {
            let spill_file = runtime
                .disk_manager
                .create_tmp_file("shuffle writer spill")?;
            let spill_data = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(spill_file.path())
                .map_err(|e| {
                    DataFusionError::Execution(format!("Error occurred while spilling {e}"))
                })?;
            self.spill_file = Some(SpillFile {
                temp_file: spill_file,
                file: spill_data,
            });
        }
        Ok(())
    }

    /// Spill the in-memory buffer to a temp file on disk, returning bytes spilled.
    pub(crate) fn spill_buffer(
        &mut self,
        runtime: &RuntimeEnv,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<usize> {
        let buffer = self
            .buf_writer
            .drain_buffer(encode_time, write_time, coalesce_time)?;
        if buffer.is_empty() {
            return Ok(0);
        }
        self.ensure_spill_file_created(runtime)?;
        let mut write_timer = write_time.timer();
        let file = &mut self.spill_file.as_mut().unwrap().file;
        file.write_all(&buffer)?;
        write_timer.stop();
        Ok(buffer.len())
    }

    /// Flush the coalescer and return remaining in-memory buffer contents for final write.
    pub(crate) fn drain_buffer(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<Vec<u8>> {
        self.buf_writer
            .drain_buffer(encode_time, write_time, coalesce_time)
    }

    pub(crate) fn path(&self) -> Option<&std::path::Path> {
        self.spill_file
            .as_ref()
            .map(|spill_file| spill_file.temp_file.path())
    }

    #[cfg(test)]
    pub(crate) fn has_spill_file(&self) -> bool {
        self.spill_file.is_some()
    }
}
