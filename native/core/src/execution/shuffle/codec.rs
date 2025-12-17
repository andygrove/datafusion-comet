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

/// Magic bytes for shuffle file header (8 bytes)
const SHUFFLE_HEADER_MAGIC: &[u8; 8] = b"CMT_STRT";
/// Magic bytes for shuffle file footer (8 bytes)
const SHUFFLE_FOOTER_MAGIC: &[u8; 8] = b"CMT_FINI";

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
    /// Pre-computed batch header: [8 bytes reserved for length] + [8 bytes field_count]
    batch_header_bytes: Vec<u8>,
    /// Codec identifier (4 bytes) written once in file header
    codec_bytes: [u8; 4],
}

impl ShuffleBlockWriter {
    pub fn try_new(schema: &Schema, codec: CompressionCodec) -> Result<Self> {
        let batch_header_bytes = Vec::with_capacity(16);
        let mut cursor = Cursor::new(batch_header_bytes);

        // leave space for compressed message length
        cursor.seek_relative(8)?;

        // write number of columns because JVM side needs to know how many addresses to allocate
        let field_count = schema.fields().len();
        cursor.write_all(&field_count.to_le_bytes())?;

        let batch_header_bytes = cursor.into_inner();

        // codec identifier to be written once in file header
        let codec_bytes: [u8; 4] = match &codec {
            CompressionCodec::Snappy => *b"SNAP",
            CompressionCodec::Lz4Frame => *b"LZ4_",
            CompressionCodec::Zstd(_) => *b"ZSTD",
            CompressionCodec::None => *b"NONE",
        };

        Ok(Self {
            codec,
            batch_header_bytes,
            codec_bytes,
        })
    }

    /// Writes the shuffle file header: magic bytes (8) + codec (4).
    /// This should be called once before writing any batches.
    /// Returns the number of bytes written (always 12).
    pub fn write_header<W: Write>(&self, output: &mut W) -> Result<usize> {
        output.write_all(SHUFFLE_HEADER_MAGIC)?;
        output.write_all(&self.codec_bytes)?;
        Ok(SHUFFLE_HEADER_MAGIC.len() + self.codec_bytes.len())
    }

    /// Writes the shuffle file footer magic bytes.
    /// This should be called once after writing all batches.
    /// Returns the number of bytes written (always 8).
    pub fn write_footer<W: Write>(&self, output: &mut W) -> Result<usize> {
        output.write_all(SHUFFLE_FOOTER_MAGIC)?;
        Ok(SHUFFLE_FOOTER_MAGIC.len())
    }

    /// Writes given record batch as Arrow IPC bytes into given writer.
    /// Returns number of bytes written.
    pub fn write_batch<W: Write + Seek>(
        &self,
        batch: &RecordBatch,
        output: &mut W,
        ipc_time: &Time,
    ) -> Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }

        let mut timer = ipc_time.timer();
        let start_pos = output.stream_position()?;

        // write batch header (length placeholder + field_count)
        output.write_all(&self.batch_header_bytes)?;

        let output = match &self.codec {
            CompressionCodec::None => {
                let mut arrow_writer = StreamWriter::try_new(output, &batch.schema())?;
                arrow_writer.write(batch)?;
                arrow_writer.finish()?;
                arrow_writer.into_inner()?
            }
            CompressionCodec::Lz4Frame => {
                let mut wtr = lz4_flex::frame::FrameEncoder::new(output);
                let mut arrow_writer = StreamWriter::try_new(&mut wtr, &batch.schema())?;
                arrow_writer.write(batch)?;
                arrow_writer.finish()?;
                wtr.finish().map_err(|e| {
                    DataFusionError::Execution(format!("lz4 compression error: {e}"))
                })?
            }

            CompressionCodec::Zstd(level) => {
                let encoder = zstd::Encoder::new(output, *level)?;
                let mut arrow_writer = StreamWriter::try_new(encoder, &batch.schema())?;
                arrow_writer.write(batch)?;
                arrow_writer.finish()?;
                let zstd_encoder = arrow_writer.into_inner()?;
                zstd_encoder.finish()?
            }

            CompressionCodec::Snappy => {
                let mut wtr = snap::write::FrameEncoder::new(output);
                let mut arrow_writer = StreamWriter::try_new(&mut wtr, &batch.schema())?;
                arrow_writer.write(batch)?;
                arrow_writer.finish()?;
                wtr.into_inner().map_err(|e| {
                    DataFusionError::Execution(format!("snappy compression error: {e}"))
                })?
            }
        };

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
}

/// Size of the length field at the start of the batch header
const LENGTH_SIZE: usize = 8;
/// Size of the field count in the batch header
const FIELD_COUNT_SIZE: usize = 8;
/// Size of the codec identifier in the file header (after magic)
const CODEC_SIZE: usize = 4;

pub struct ShuffleBlockReader<R: std::io::Read> {
    reader: R,
    /// Codec read from file header, used for all batches
    codec: [u8; CODEC_SIZE],
}

impl<R: std::io::Read> ShuffleBlockReader<R> {
    /// Creates a new ShuffleBlockReader, verifies the header magic bytes, and reads the codec.
    pub fn try_new(mut reader: R) -> Result<Self> {
        // Read and verify header magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != SHUFFLE_HEADER_MAGIC {
            return Err(DataFusionError::Execution(format!(
                "Invalid shuffle file header: expected {:?}, got {:?}",
                SHUFFLE_HEADER_MAGIC, magic
            )));
        }

        // Read codec (4 bytes immediately after magic)
        let mut codec = [0u8; CODEC_SIZE];
        reader.read_exact(&mut codec)?;

        Ok(Self { reader, codec })
    }

    /// Reads and verifies a section header magic and codec.
    /// Returns true if header was found, false if EOF.
    fn read_section_header(&mut self) -> Result<bool> {
        let mut magic = [0u8; 8];
        match self.reader.read_exact(&mut magic) {
            Ok(()) => {
                if &magic != SHUFFLE_HEADER_MAGIC {
                    return Err(DataFusionError::Execution(format!(
                        "Invalid shuffle section header: expected {:?}, got {:?}",
                        SHUFFLE_HEADER_MAGIC, magic
                    )));
                }
                // Read codec for this section (4 bytes after magic)
                self.reader.read_exact(&mut self.codec)?;
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    pub fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        // Read the first 8 bytes - could be batch header or footer magic
        let mut first_bytes = [0u8; LENGTH_SIZE];
        match self.reader.read_exact(&mut first_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }

        // Check if this is the footer magic (end of current section)
        if &first_bytes == SHUFFLE_FOOTER_MAGIC {
            // Try to read the next section header
            if self.read_section_header()? {
                // There's another section, try to read the next batch
                return self.next_batch();
            }
            // No more sections, we're done
            return Ok(None);
        }

        // Parse ipc_length (8 bytes, little-endian)
        let ipc_length = u64::from_le_bytes(first_bytes) as usize;

        // Read the rest of the batch header (field_count only, codec is in file header)
        let mut field_count_bytes = [0u8; FIELD_COUNT_SIZE];
        self.reader.read_exact(&mut field_count_bytes)?;

        // Parse field_count (8 bytes, little-endian) - read but not currently used
        let _field_count = usize::from_le_bytes(field_count_bytes);

        // Calculate compressed data length (ipc_length includes field_count + data)
        let compressed_data_len = ipc_length - FIELD_COUNT_SIZE;

        // Read the compressed data
        let mut compressed_data = vec![0u8; compressed_data_len];
        self.reader.read_exact(&mut compressed_data)?;

        // Decompress and read the IPC batch based on codec from file header
        let batch = match &self.codec {
            b"SNAP" => {
                let decoder = snap::read::FrameDecoder::new(compressed_data.as_slice());
                let mut reader =
                    unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
                reader.next().ok_or_else(|| {
                    DataFusionError::Execution("No batch in shuffle block".to_string())
                })??
            }
            b"LZ4_" => {
                let decoder = lz4_flex::frame::FrameDecoder::new(compressed_data.as_slice());
                let mut reader =
                    unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
                reader.next().ok_or_else(|| {
                    DataFusionError::Execution("No batch in shuffle block".to_string())
                })??
            }
            b"ZSTD" => {
                let decoder = zstd::Decoder::new(compressed_data.as_slice())?;
                let mut reader =
                    unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
                reader.next().ok_or_else(|| {
                    DataFusionError::Execution("No batch in shuffle block".to_string())
                })??
            }
            b"NONE" => {
                let mut reader = unsafe {
                    StreamReader::try_new(compressed_data.as_slice(), None)?
                        .with_skip_validation(true)
                };
                reader.next().ok_or_else(|| {
                    DataFusionError::Execution("No batch in shuffle block".to_string())
                })??
            }
            other => {
                return Err(DataFusionError::Execution(format!(
                    "Invalid compression codec in shuffle block: {:?}",
                    other
                )));
            }
        };

        Ok(Some(batch))
    }

    /// Consumes the reader and returns the inner reader
    pub fn into_inner(self) -> R {
        self.reader
    }
}

/// Decompresses and reads an IPC batch using the specified codec.
///
/// # Arguments
/// * `codec` - 4-byte codec identifier (e.g., b"SNAP", b"LZ4_", b"ZSTD", b"NONE")
/// * `bytes` - The compressed IPC data (without codec prefix)
pub fn read_ipc_compressed(codec: &[u8; 4], bytes: &[u8]) -> Result<RecordBatch> {
    match codec {
        b"SNAP" => {
            let decoder = snap::read::FrameDecoder::new(bytes);
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"LZ4_" => {
            let decoder = lz4_flex::frame::FrameDecoder::new(bytes);
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"ZSTD" => {
            let decoder = zstd::Decoder::new(bytes)?;
            let mut reader =
                unsafe { StreamReader::try_new(decoder, None)?.with_skip_validation(true) };
            reader.next().unwrap().map_err(|e| e.into())
        }
        b"NONE" => {
            let mut reader =
                unsafe { StreamReader::try_new(bytes, None)?.with_skip_validation(true) };
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

#[cfg(test)]
mod test {
    use crate::execution::shuffle::CompressionCodec;
    use crate::execution::shuffle::ShuffleBlockReader;
    use crate::execution::shuffle::ShuffleBlockWriter;
    use arrow::array::{RecordBatch, StringBuilder};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::DataFusionError;
    use datafusion::physical_plan::metrics::Time;
    use std::io::Cursor;
    use std::sync::Arc;

    #[test]
    fn shuffle_file_roundtrip() -> Result<(), DataFusionError> {
        let batch = create_batch(8192);
        // create multiple block writers to simulate combining multiple spill files
        let t = Time::new();
        let mut output = vec![];
        let mut cursor = Cursor::new(&mut output);
        for _ in 0..5 {
            let schema = batch.schema();
            let writer = ShuffleBlockWriter::try_new(schema.as_ref(), CompressionCodec::Zstd(3))?;
            // Write header before batches
            writer.write_header(&mut cursor)?;
            for _ in 0..10 {
                writer.write_batch(&batch, &mut cursor, &t)?;
            }
            // Write footer after batches
            writer.write_footer(&mut cursor)?;
        }
        // Read all batches using a single ShuffleBlockReader
        // The reader handles multiple sections (header/footer pairs) seamlessly
        let mut reader = ShuffleBlockReader::try_new(Cursor::new(&output))?;
        let mut batch_count = 0;
        while let Some(read_batch) = reader.next_batch()? {
            assert_eq!(read_batch.num_rows(), batch.num_rows());
            assert_eq!(read_batch.num_columns(), batch.num_columns());
            assert_eq!(read_batch.schema(), batch.schema());
            batch_count += 1;
        }
        assert_eq!(batch_count, 50); // 5 writers * 10 batches each
        Ok(())
    }

    fn create_batch(batch_size: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, true)]));
        let mut b = StringBuilder::new();
        for i in 0..batch_size {
            b.append_value(format!("{i}"));
        }
        let array = b.finish();
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
    }
}
