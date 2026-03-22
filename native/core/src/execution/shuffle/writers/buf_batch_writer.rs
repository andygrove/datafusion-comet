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

use crate::execution::shuffle::ShuffleBlockWriter;
use arrow::array::RecordBatch;
use arrow::compute::kernels::coalesce::BatchCoalescer;
use datafusion::physical_plan::metrics::Time;
use std::borrow::Borrow;
use std::io::{Cursor, Seek, SeekFrom, Write};

/// Write batches to writer while using a buffer to avoid frequent system calls.
/// The record batches were first written by ShuffleBlockWriter into an internal buffer.
/// Once the buffer exceeds the max size, the buffer will be flushed to the writer.
///
/// Small batches are coalesced using Arrow's [`BatchCoalescer`] with a memory-based
/// flush threshold. The coalescer uses `usize::MAX` as its row target (so it never
/// auto-flushes by row count), and instead we track accumulated memory and call
/// `finish_buffered_batch()` when the target `block_size` is reached. This produces
/// fewer, larger IPC blocks while using efficient incremental copies.
pub(crate) struct BufBatchWriter<S: Borrow<ShuffleBlockWriter>, W: Write> {
    shuffle_block_writer: S,
    writer: W,
    buffer: Vec<u8>,
    buffer_max_size: usize,
    /// Coalesces batches incrementally using pre-allocated in-progress arrays.
    /// Lazily initialized on first write to capture the schema.
    coalescer: Option<BatchCoalescer>,
    /// Estimated memory accumulated in the coalescer since last flush
    pending_memory: usize,
    /// Target memory size for each IPC block
    block_size: usize,
}

impl<S: Borrow<ShuffleBlockWriter>, W: Write> BufBatchWriter<S, W> {
    pub(crate) fn new(
        shuffle_block_writer: S,
        writer: W,
        buffer_max_size: usize,
        block_size: usize,
    ) -> Self {
        Self {
            shuffle_block_writer,
            writer,
            buffer: vec![],
            buffer_max_size,
            coalescer: None,
            pending_memory: 0,
            block_size,
        }
    }

    pub(crate) fn write(
        &mut self,
        batch: RecordBatch,
        encode_time: &Time,
        write_time: &Time,
    ) -> datafusion::common::Result<usize> {
        let batch_memory = batch.get_array_memory_size();

        let coalescer = self.coalescer.get_or_insert_with(|| {
            // Large row target so the coalescer never auto-flushes by row count;
            // we control flushing via the memory-based threshold instead.
            BatchCoalescer::new(batch.schema(), 1_000_000_000)
        });
        coalescer.push_batch(batch)?;
        self.pending_memory += batch_memory;

        let mut bytes_written = 0;
        if self.pending_memory >= self.block_size {
            bytes_written += self.flush_coalescer(encode_time, write_time)?;
        }
        Ok(bytes_written)
    }

    /// Flush the coalescer's buffered data and write it as a single IPC block.
    fn flush_coalescer(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
    ) -> datafusion::common::Result<usize> {
        let Some(coalescer) = &mut self.coalescer else {
            return Ok(0);
        };
        coalescer.finish_buffered_batch()?;

        let mut completed = Vec::new();
        while let Some(batch) = coalescer.next_completed_batch() {
            completed.push(batch);
        }
        self.pending_memory = 0;

        let mut bytes_written = 0;
        for batch in &completed {
            bytes_written += self.write_batch_to_buffer(batch, encode_time, write_time)?;
        }
        Ok(bytes_written)
    }

    /// Serialize a single batch into the byte buffer, flushing to the writer if needed.
    fn write_batch_to_buffer(
        &mut self,
        batch: &RecordBatch,
        encode_time: &Time,
        write_time: &Time,
    ) -> datafusion::common::Result<usize> {
        let mut cursor = Cursor::new(&mut self.buffer);
        cursor.seek(SeekFrom::End(0))?;
        let bytes_written =
            self.shuffle_block_writer
                .borrow()
                .write_batch(batch, &mut cursor, encode_time)?;
        let pos = cursor.position();
        if pos >= self.buffer_max_size as u64 {
            let mut write_timer = write_time.timer();
            self.writer.write_all(&self.buffer)?;
            write_timer.stop();
            self.buffer.clear();
        }
        Ok(bytes_written)
    }

    pub(crate) fn flush(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
    ) -> datafusion::common::Result<()> {
        // Flush any remaining data in the coalescer
        self.flush_coalescer(encode_time, write_time)?;

        // Flush the byte buffer to the underlying writer
        let mut write_timer = write_time.timer();
        if !self.buffer.is_empty() {
            self.writer.write_all(&self.buffer)?;
        }
        self.writer.flush()?;
        write_timer.stop();
        self.buffer.clear();
        Ok(())
    }
}

impl<S: Borrow<ShuffleBlockWriter>, W: Write + Seek> BufBatchWriter<S, W> {
    pub(crate) fn writer_stream_position(&mut self) -> datafusion::common::Result<u64> {
        self.writer.stream_position().map_err(Into::into)
    }
}
