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
use crate::execution::shuffle::partitioners::ShufflePartitioner;
use crate::execution::shuffle::{ipc_write_options, CompressionCodec};
use arrow::array::RecordBatch;
use arrow::compute::kernels::coalesce::BatchCoalescer;
use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;
use datafusion::common::DataFusionError;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use tokio::time::Instant;

/// A partitioner that writes all shuffle data to a single file as a single Arrow IPC stream.
pub(crate) struct SinglePartitionShufflePartitioner {
    ipc_writer: StreamWriter<File>,
    output_index_path: String,
    coalescer: Option<BatchCoalescer>,
    metrics: ShufflePartitionerMetrics,
    batch_size: usize,
}

impl SinglePartitionShufflePartitioner {
    pub(crate) fn try_new(
        output_data_path: String,
        output_index_path: String,
        schema: SchemaRef,
        metrics: ShufflePartitionerMetrics,
        batch_size: usize,
        codec: CompressionCodec,
    ) -> datafusion::common::Result<Self> {
        let ipc_opts = ipc_write_options(&codec)?;

        let output_data_file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .truncate(true)
            .open(output_data_path)?;

        let ipc_writer =
            StreamWriter::try_new_with_options(output_data_file, &schema, ipc_opts)?;

        Ok(Self {
            ipc_writer,
            output_index_path,
            coalescer: None,
            metrics,
            batch_size,
        })
    }

    /// Flush the coalescer's remaining buffered rows to the IPC writer.
    fn flush_coalescer(&mut self) -> datafusion::common::Result<()> {
        let mut remaining = Vec::new();
        if let Some(coalescer) = &mut self.coalescer {
            coalescer.finish_buffered_batch()?;
            while let Some(batch) = coalescer.next_completed_batch() {
                remaining.push(batch);
            }
        }
        if !remaining.is_empty() {
            let mut timer = self.metrics.encode_time.timer();
            for batch in &remaining {
                self.ipc_writer.write(batch)?;
            }
            timer.stop();
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl ShufflePartitioner for SinglePartitionShufflePartitioner {
    async fn insert_batch(&mut self, batch: RecordBatch) -> datafusion::common::Result<()> {
        let start_time = Instant::now();
        let num_rows = batch.num_rows();

        if num_rows > 0 {
            self.metrics.data_size.add(batch.get_array_memory_size());
            self.metrics.baseline.record_output(num_rows);

            if num_rows >= self.batch_size {
                // Large batch: flush coalescer first, then write directly
                self.flush_coalescer()?;
                let mut timer = self.metrics.encode_time.timer();
                self.ipc_writer.write(&batch)?;
                timer.stop();
            } else {
                // Small batch: push to coalescer
                let coalescer = self
                    .coalescer
                    .get_or_insert_with(|| BatchCoalescer::new(batch.schema(), self.batch_size));
                coalescer.push_batch(batch)?;

                // Drain completed batches to the IPC writer
                let mut completed = Vec::new();
                while let Some(batch) = coalescer.next_completed_batch() {
                    completed.push(batch);
                }
                if !completed.is_empty() {
                    let mut timer = self.metrics.encode_time.timer();
                    for batch in &completed {
                        self.ipc_writer.write(batch)?;
                    }
                    timer.stop();
                }
            }
        }

        self.metrics.input_batches.add(1);
        self.metrics
            .baseline
            .elapsed_compute()
            .add_duration(start_time.elapsed());
        Ok(())
    }

    fn shuffle_write(&mut self) -> datafusion::common::Result<()> {
        let start_time = Instant::now();

        // Flush coalescer and finish the IPC stream
        self.flush_coalescer()?;
        {
            let mut timer = self.metrics.encode_time.timer();
            self.ipc_writer.finish()?;
            timer.stop();
        }

        // Write index file. It should only contain 2 entries: 0 and the total number of bytes written
        let data_file_length = self.ipc_writer.get_mut().stream_position()?;
        let index_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(self.output_index_path.clone())
            .map_err(|e| DataFusionError::Execution(format!("shuffle write error: {e:?}")))?;
        let mut index_buf_writer = BufWriter::new(index_file);
        for offset in [0, data_file_length] {
            index_buf_writer.write_all(&(offset as i64).to_le_bytes()[..])?;
        }
        index_buf_writer.flush()?;

        self.metrics
            .baseline
            .elapsed_compute()
            .add_duration(start_time.elapsed());
        Ok(())
    }
}
