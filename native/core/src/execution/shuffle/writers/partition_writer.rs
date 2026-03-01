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
use crate::execution::shuffle::writers::buf_batch_writer::BufBatchWriter;
use crate::execution::shuffle::ShuffleBlockWriter;
use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use std::fs::{File, OpenOptions};

pub(crate) struct PartitionWriter {
    temp_file: Option<RefCountedTempFile>,
    buf_writer: Option<BufBatchWriter<ShuffleBlockWriter, File>>,
    shuffle_block_writer: ShuffleBlockWriter,
    write_buffer_size: usize,
    batch_size: usize,
}

impl PartitionWriter {
    pub(crate) fn try_new(
        shuffle_block_writer: ShuffleBlockWriter,
        write_buffer_size: usize,
        batch_size: usize,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            temp_file: None,
            buf_writer: None,
            shuffle_block_writer,
            write_buffer_size,
            batch_size,
        })
    }

    /// Write a batch to this partition's spill file, lazily creating the file and writer.
    pub(crate) fn write_batch(
        &mut self,
        batch: &RecordBatch,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<usize> {
        if self.buf_writer.is_none() {
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
            self.buf_writer = Some(BufBatchWriter::new(
                self.shuffle_block_writer.clone(),
                file,
                self.write_buffer_size,
                self.batch_size,
            ));
        }
        self.buf_writer.as_mut().unwrap().write(
            batch,
            &metrics.encode_time,
            &metrics.write_time,
        )
    }

    /// Flush the persistent BufBatchWriter (coalescer + byte buffer).
    pub(crate) fn flush(
        &mut self,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        if let Some(writer) = self.buf_writer.as_mut() {
            writer.flush(&metrics.encode_time, &metrics.write_time)?;
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
