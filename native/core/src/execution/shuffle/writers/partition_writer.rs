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
use crate::execution::shuffle::ShuffleBlockWriter;
use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use std::fs::{File, OpenOptions};

struct SpillFile {
    temp_file: RefCountedTempFile,
    file: File,
}

pub(crate) struct PartitionWriter {
    /// Spill file for intermediate shuffle output for this partition. Each spill event
    /// will append to this file and the contents will be copied to the shuffle file at
    /// the end of processing.
    spill_file: Option<SpillFile>,
    /// Writer that performs encoding and compression
    shuffle_block_writer: ShuffleBlockWriter,
}

impl PartitionWriter {
    pub(crate) fn try_new(
        shuffle_block_writer: ShuffleBlockWriter,
    ) -> datafusion::common::Result<Self> {
        Ok(Self {
            spill_file: None,
            shuffle_block_writer,
        })
    }

    fn ensure_spill_file_created(
        &mut self,
        runtime: &RuntimeEnv,
    ) -> datafusion::common::Result<()> {
        if self.spill_file.is_none() {
            // Spill file is not yet created, create it
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

    /// Write a single batch directly to the spill file without coalescing.
    /// Use this when the caller already produces well-sized batches (e.g. from
    /// the scatter kernel's PartitionBuffer which flushes at batch_size).
    pub(crate) fn spill_direct(
        &mut self,
        batch: &RecordBatch,
        runtime: &RuntimeEnv,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }
        self.ensure_spill_file_created(runtime)?;
        let file = &mut self.spill_file.as_mut().unwrap().file;
        self.shuffle_block_writer
            .write_batch(batch, file, &metrics.encode_time)
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
