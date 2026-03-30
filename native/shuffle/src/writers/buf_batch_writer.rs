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
use arrow::array::RecordBatch;
use arrow::compute::kernels::coalesce::BatchCoalescer;
use datafusion::physical_plan::metrics::Time;
use std::borrow::Borrow;
use std::io::{Cursor, Seek, SeekFrom, Write};

/// Write batches to writer while using a buffer to avoid frequent system calls.
/// The record batches were first written by ShuffleBlockWriter into an internal buffer.
/// Once the buffer exceeds the max size, the buffer will be flushed to the writer.
///
/// Small batches are coalesced using Arrow's [`BatchCoalescer`] before serialization,
/// producing exactly `batch_size`-row output batches to reduce per-block IPC schema overhead.
/// The coalescer is lazily initialized on the first write.
pub(crate) struct BufBatchWriter<S: Borrow<ShuffleBlockWriter>, W: Write> {
    shuffle_block_writer: S,
    writer: W,
    buffer: Vec<u8>,
    buffer_max_size: usize,
    /// Coalesces small batches into target_batch_size before serialization.
    /// Lazily initialized on first write to capture the schema.
    coalescer: Option<BatchCoalescer>,
    /// Target batch size for coalescing
    batch_size: usize,
    /// Estimated memory held by unencoded batches in the coalescer
    coalescer_buffered_bytes: usize,
}

impl<S: Borrow<ShuffleBlockWriter>, W: Write> BufBatchWriter<S, W> {
    pub(crate) fn new(
        shuffle_block_writer: S,
        writer: W,
        buffer_max_size: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            shuffle_block_writer,
            writer,
            buffer: vec![],
            buffer_max_size,
            coalescer: None,
            batch_size,
            coalescer_buffered_bytes: 0,
        }
    }

    pub(crate) fn write(
        &mut self,
        batch: &RecordBatch,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<usize> {
        let input_mem = batch.get_array_memory_size();

        let mut coalesce_timer = coalesce_time.timer();
        let coalescer = self
            .coalescer
            .get_or_insert_with(|| BatchCoalescer::new(batch.schema(), self.batch_size));
        coalescer.push_batch(batch.clone())?;

        // Drain completed batches into a local vec so the coalescer borrow ends
        // before we call write_batch_to_buffer (which borrows &mut self).
        let mut completed = Vec::new();
        while let Some(batch) = coalescer.next_completed_batch() {
            completed.push(batch);
        }
        coalesce_timer.stop();

        // Track coalescer memory: input added, completed batches released
        self.coalescer_buffered_bytes += input_mem;
        for batch in &completed {
            self.coalescer_buffered_bytes = self
                .coalescer_buffered_bytes
                .saturating_sub(batch.get_array_memory_size());
        }

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

    /// Flush remaining rows from the coalescer into the internal byte buffer.
    fn flush_coalescer(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<()> {
        let mut coalesce_timer = coalesce_time.timer();
        let mut remaining = Vec::new();
        if let Some(coalescer) = &mut self.coalescer {
            coalescer.finish_buffered_batch()?;
            while let Some(batch) = coalescer.next_completed_batch() {
                remaining.push(batch);
            }
        }
        coalesce_timer.stop();
        for batch in &remaining {
            self.write_batch_to_buffer(batch, encode_time, write_time)?;
        }
        self.coalescer_buffered_bytes = 0;
        Ok(())
    }

    pub(crate) fn flush(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<()> {
        self.flush_coalescer(encode_time, write_time, coalesce_time)?;

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

    pub(crate) fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Estimated memory held by unencoded batches in the coalescer.
    pub(crate) fn coalescer_buffered_bytes(&self) -> usize {
        self.coalescer_buffered_bytes
    }

    /// Flush the coalescer and return the internal buffer contents.
    pub(crate) fn drain_buffer(
        &mut self,
        encode_time: &Time,
        write_time: &Time,
        coalesce_time: &Time,
    ) -> datafusion::common::Result<Vec<u8>> {
        self.flush_coalescer(encode_time, write_time, coalesce_time)?;
        Ok(std::mem::take(&mut self.buffer))
    }
}

impl<S: Borrow<ShuffleBlockWriter>, W: Write + Seek> BufBatchWriter<S, W> {
    pub(crate) fn writer_stream_position(&mut self) -> datafusion::common::Result<u64> {
        self.writer.stream_position().map_err(Into::into)
    }
}
