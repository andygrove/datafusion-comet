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

use crate::errors::{CometError, CometResult};
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use bytes::Buf;
use crc32fast::Hasher;
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::physical_plan::metrics::Time;
use simd_adler32::Adler32;
use std::io::{Cursor, Seek, SeekFrom, Write};

#[derive(Debug, Clone)]
pub enum CompressionCodec {
    None,
    Lz4Frame,
    Zstd(i32),
    Snappy,
}

#[derive(Clone)]
pub struct ShuffleBlockWriter {
    codec: CompressionCodec,
    header_bytes: Vec<u8>,
    schema_written: bool,
}

impl ShuffleBlockWriter {
    pub fn try_new(schema: &Schema, codec: CompressionCodec) -> Result<Self> {
        let header_bytes = Vec::with_capacity(20);
        let mut cursor = Cursor::new(header_bytes);

        // leave space for compressed message length
        cursor.seek_relative(8)?;

        // write number of columns because JVM side needs to know how many addresses to allocate
        let field_count = schema.fields().len();
        cursor.write_all(&field_count.to_le_bytes())?;

        // write compression codec to header
        let codec_header = match &codec {
            CompressionCodec::Snappy => b"SNAP",
            CompressionCodec::Lz4Frame => b"LZ4_",
            CompressionCodec::Zstd(_) => b"ZSTD",
            CompressionCodec::None => b"NONE",
        };
        cursor.write_all(codec_header)?;

        let header_bytes = cursor.into_inner();

        Ok(Self {
            codec,
            header_bytes,
            schema_written: false,
        })
    }

    /// Writes given record batch as Arrow IPC bytes into given writer.
    /// First call writes codec + schema + batch. Subsequent calls write just batch.
    /// Returns number of bytes written.
    pub fn write_batch<W: Write + Seek>(
        &mut self,
        batch: &RecordBatch,
        output: &mut W,
        ipc_time: &Time,
    ) -> Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }

        let mut timer = ipc_time.timer();
        let start_pos = output.stream_position()?;

        // write header
        output.write_all(&self.header_bytes)?;

        // Create compressed data in memory first
        let compressed_data = if !self.schema_written {
            // First batch: write schema + batch
            self.create_compressed_data_with_schema(batch)?
        } else {
            // Subsequent batches: write just batch data (for now, still includes schema)
            // TODO: optimize to exclude schema once reader is updated
            self.create_compressed_data_with_schema(batch)?
        };

        // Write compressed data to output
        output.write_all(&compressed_data)?;

        if !self.schema_written {
            self.schema_written = true;
        }

        // fill ipc length
        let end_pos = output.stream_position()?;
        let ipc_length = end_pos - start_pos - 8;
        let max_size = i32::MAX as u64;
        if ipc_length > max_size {
            return Err(DataFusionError::Execution(format!(
                "Shuffle block size {ipc_length} exceeds maximum size of {max_size}. \
                Try reducing batch size or increasing compression level"
            )));
        }

        // fill ipc length
        output.seek(SeekFrom::Start(start_pos))?;
        output.write_all(&ipc_length.to_le_bytes())?;
        output.seek(SeekFrom::Start(end_pos))?;

        timer.stop();

        Ok((end_pos - start_pos) as usize)
    }

    fn create_compressed_data_with_schema(&self, batch: &RecordBatch) -> Result<Vec<u8>> {
        match &self.codec {
            CompressionCodec::None => {
                let mut temp_buf = Vec::new();
                let mut arrow_writer = StreamWriter::try_new(&mut temp_buf, &batch.schema())?;
                arrow_writer.write(batch)?;
                arrow_writer.finish()?;
                Ok(temp_buf)
            }
            CompressionCodec::Lz4Frame => {
                let mut temp_buf = Vec::new();
                {
                    let mut wtr = lz4_flex::frame::FrameEncoder::new(&mut temp_buf);
                    let mut arrow_writer = StreamWriter::try_new(&mut wtr, &batch.schema())?;
                    arrow_writer.write(batch)?;
                    arrow_writer.finish()?;
                    wtr.finish().map_err(|e| {
                        DataFusionError::Execution(format!("lz4 compression error: {e}"))
                    })?;
                }
                Ok(temp_buf)
            }
            CompressionCodec::Zstd(level) => {
                let mut temp_buf = Vec::new();
                {
                    let mut encoder = zstd::Encoder::new(&mut temp_buf, *level)?;
                    let mut arrow_writer = StreamWriter::try_new(&mut encoder, &batch.schema())?;
                    arrow_writer.write(batch)?;
                    arrow_writer.finish()?;
                    encoder.finish()?;
                }
                Ok(temp_buf)
            }
            CompressionCodec::Snappy => {
                let mut temp_buf = Vec::new();
                {
                    let mut wtr = snap::write::FrameEncoder::new(&mut temp_buf);
                    let mut arrow_writer = StreamWriter::try_new(&mut wtr, &batch.schema())?;
                    arrow_writer.write(batch)?;
                    arrow_writer.finish()?;
                    wtr.into_inner().map_err(|e| {
                        DataFusionError::Execution(format!("snappy compression error: {e}"))
                    })?;
                }
                Ok(temp_buf)
            }
        }
    }
}

pub fn read_ipc_compressed(bytes: &[u8]) -> Result<RecordBatch> {
    match &bytes[0..4] {
        b"SNAP" => {
            let decoder = snap::read::FrameDecoder::new(&bytes[4..]);
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"LZ4_" => {
            let decoder = lz4_flex::frame::FrameDecoder::new(&bytes[4..]);
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"ZSTD" => {
            let decoder = zstd::Decoder::new(&bytes[4..])?;
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"NONE" => {
            let mut reader =
                unsafe { StreamReader::try_new(&bytes[4..], None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        other => Err(DataFusionError::Execution(format!(
            "Failed to decode batch: invalid compression codec: {other:?}"
        ))),
    }
}

/// Checksum algorithms for writing IPC bytes.
#[derive(Clone)]
pub(crate) enum Checksum {
    /// CRC32 checksum algorithm.
    CRC32(Hasher),
    /// Adler32 checksum algorithm.
    Adler32(Adler32),
}

impl Checksum {
    pub(crate) fn try_new(algo: i32, initial_opt: Option<u32>) -> CometResult<Self> {
        match algo {
            0 => {
                let hasher = if let Some(initial) = initial_opt {
                    Hasher::new_with_initial(initial)
                } else {
                    Hasher::new()
                };
                Ok(Checksum::CRC32(hasher))
            }
            1 => {
                let hasher = if let Some(initial) = initial_opt {
                    // Note that Adler32 initial state is not zero.
                    // i.e., `Adler32::from_checksum(0)` is not the same as `Adler32::new()`.
                    Adler32::from_checksum(initial)
                } else {
                    Adler32::new()
                };
                Ok(Checksum::Adler32(hasher))
            }
            _ => Err(CometError::Internal(
                "Unsupported checksum algorithm".to_string(),
            )),
        }
    }

    pub(crate) fn update(&mut self, cursor: &mut Cursor<&mut Vec<u8>>) -> CometResult<()> {
        match self {
            Checksum::CRC32(hasher) => {
                std::io::Seek::seek(cursor, SeekFrom::Start(0))?;
                hasher.update(cursor.chunk());
                Ok(())
            }
            Checksum::Adler32(hasher) => {
                std::io::Seek::seek(cursor, SeekFrom::Start(0))?;
                hasher.write(cursor.chunk());
                Ok(())
            }
        }
    }

    pub(crate) fn finalize(self) -> u32 {
        match self {
            Checksum::CRC32(hasher) => hasher.finalize(),
            Checksum::Adler32(hasher) => hasher.finish(),
        }
    }
}
