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
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::runtime_env::RuntimeEnv;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// A `Write` wrapper that defers temp file creation until the first actual write.
/// This allows `BufBatchWriter` to be constructed eagerly per partition without
/// creating spill files for partitions that never receive data.
pub(crate) struct LazySpillFile {
    runtime: Arc<RuntimeEnv>,
    inner: Option<SpillFileInner>,
}

struct SpillFileInner {
    temp_file: RefCountedTempFile,
    file: File,
}

impl LazySpillFile {
    fn new(runtime: Arc<RuntimeEnv>) -> Self {
        Self {
            runtime,
            inner: None,
        }
    }

    fn ensure_created(&mut self) -> std::io::Result<()> {
        if self.inner.is_none() {
            let temp_file = self
                .runtime
                .disk_manager
                .create_tmp_file("shuffle writer spill")
                .map_err(std::io::Error::other)?;
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(temp_file.path())?;
            self.inner = Some(SpillFileInner { temp_file, file });
        }
        Ok(())
    }

    pub(crate) fn path(&self) -> Option<&Path> {
        self.inner.as_ref().map(|f| f.temp_file.path())
    }

    #[cfg(test)]
    pub(crate) fn has_data(&self) -> bool {
        self.inner.is_some()
    }
}

impl Write for LazySpillFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.ensure_created()?;
        self.inner.as_mut().unwrap().file.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(inner) = &mut self.inner {
            inner.file.flush()
        } else {
            Ok(())
        }
    }
}

/// Per-partition writer that holds a persistent `BufBatchWriter` with a `BatchCoalescer`.
/// Small sub-batches (from Arrow `take`) are coalesced into `batch_size` output batches
/// before serialization, avoiding per-block IPC overhead.
pub(crate) struct PartitionWriter {
    buf_writer: BufBatchWriter<ShuffleBlockWriter, LazySpillFile>,
}

impl PartitionWriter {
    pub(crate) fn try_new(
        shuffle_block_writer: ShuffleBlockWriter,
        runtime: Arc<RuntimeEnv>,
        write_buffer_size: usize,
        batch_size: usize,
    ) -> datafusion::common::Result<Self> {
        let lazy_file = LazySpillFile::new(runtime);
        let buf_writer = BufBatchWriter::new(
            shuffle_block_writer,
            lazy_file,
            write_buffer_size,
            batch_size,
        );
        Ok(Self { buf_writer })
    }

    pub(crate) fn write(
        &mut self,
        batch: &RecordBatch,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<usize> {
        self.buf_writer
            .write(batch, &metrics.encode_time, &metrics.write_time)
    }

    pub(crate) fn flush(
        &mut self,
        metrics: &ShufflePartitionerMetrics,
    ) -> datafusion::common::Result<()> {
        self.buf_writer
            .flush(&metrics.encode_time, &metrics.write_time)
    }

    pub(crate) fn path(&self) -> Option<&Path> {
        self.buf_writer.writer().path()
    }

    #[cfg(test)]
    pub(crate) fn has_spill_file(&self) -> bool {
        self.buf_writer.writer().has_data()
    }
}
