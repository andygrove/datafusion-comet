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
use arrow::array::{ArrayData, RecordBatch};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Schema};
use bytes::Buf;
use crc32fast::Hasher;
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::physical_plan::metrics::Time;
use simd_adler32::Adler32;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::Arc;

/// CSB1 magic bytes identifying the Comet Shuffle Block v1 format.
const CSB1_MAGIC: &[u8; 4] = b"CSB1";

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
}

/// Recursively collect array node metadata and buffer data in DFS order.
///
/// For each array node, appends its null_count to `null_counts`.
/// For each buffer, appends its byte length to `buf_lengths` and
/// a reference to its data to `buf_data`. When null_count is 0,
/// the validity bitmap is omitted (length set to 0, no data appended).
fn collect_array_buffers<'a>(
    array: &'a ArrayData,
    null_counts: &mut Vec<u32>,
    buf_lengths: &mut Vec<u32>,
    buf_data: &mut Vec<&'a [u8]>,
) {
    let null_count = array.null_count() as u32;
    null_counts.push(null_count);

    // First buffer for most types is the validity bitmap
    let buffers = array.buffers();
    let has_validity = matches!(
        array.data_type(),
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Struct(_)
            | DataType::Map(_, _)
    );

    if has_validity {
        if null_count == 0 {
            // Omit validity bitmap when there are no nulls
            buf_lengths.push(0);
        } else {
            let nulls = array
                .nulls()
                .expect("null_count > 0 but no null buffer present");
            let validity_bytes = nulls.buffer().as_slice();
            buf_lengths.push(validity_bytes.len() as u32);
            buf_data.push(validity_bytes);
        }
    }

    // Append the remaining data buffers (offsets, values, etc.)
    for buf in buffers {
        let slice = buf.as_slice();
        buf_lengths.push(slice.len() as u32);
        buf_data.push(slice);
    }

    // Recurse into child arrays
    for child in array.child_data() {
        collect_array_buffers(child, null_counts, buf_lengths, buf_data);
    }
}

/// Reconstruct an ArrayData from CSB1 buffer data, walking the schema in DFS order.
///
/// Consumes entries from `null_counts`, `buf_lengths`, and `buf_data` as it
/// reconstructs each array node. The `length` parameter is the number of rows
/// for this array.
fn reconstruct_array(
    dt: &DataType,
    length: usize,
    null_counts: &[u32],
    buf_lengths: &[u32],
    buf_data: &[u8],
    node_idx: &mut usize,
    buf_idx: &mut usize,
    data_offset: &mut usize,
) -> std::result::Result<ArrayData, DataFusionError> {
    let ni = *node_idx;
    *node_idx += 1;
    let null_count = null_counts[ni] as usize;

    let has_validity = matches!(
        dt,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Struct(_)
            | DataType::Map(_, _)
    );

    // Read validity bitmap
    let null_buffer = if has_validity {
        let bi = *buf_idx;
        *buf_idx += 1;
        let len = buf_lengths[bi] as usize;
        if len == 0 {
            // No validity bitmap (null_count == 0)
            None
        } else {
            let buf = Buffer::from(&buf_data[*data_offset..*data_offset + len]);
            *data_offset += len;
            Some(buf)
        }
    } else {
        None
    };

    // Determine how many data buffers this type has (excluding validity)
    let num_data_buffers = match dt {
        DataType::Boolean => 1, // values bitmap
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => 1, // values
        DataType::Float32 | DataType::Float64 => 1, // values
        DataType::Date32 | DataType::Date64 => 1, // values
        DataType::Timestamp(_, _) => 1, // values
        DataType::Decimal128(_, _) => 1, // values
        DataType::Utf8 | DataType::Binary => 2, // offsets + data
        DataType::LargeUtf8 | DataType::LargeBinary => 2, // offsets + data
        DataType::List(_) | DataType::LargeList(_) => 1, // offsets only
        DataType::Struct(_) => 0, // no data buffers
        DataType::Map(_, _) => 1, // offsets only
        other => {
            return Err(DataFusionError::Execution(format!(
                "Unsupported data type in CSB1 reader: {other:?}"
            )));
        }
    };

    // Read data buffers
    let mut buffers = Vec::with_capacity(num_data_buffers);
    for _ in 0..num_data_buffers {
        let bi = *buf_idx;
        *buf_idx += 1;
        let len = buf_lengths[bi] as usize;
        let buf = Buffer::from(&buf_data[*data_offset..*data_offset + len]);
        *data_offset += len;
        buffers.push(buf);
    }

    // Reconstruct child arrays
    let child_data = match dt {
        DataType::List(field) | DataType::LargeList(field) => {
            // Child length = last offset value
            let offsets_buf = &buffers[0];
            let child_length = if length == 0 {
                0
            } else {
                let offset_bytes = &offsets_buf.as_slice();
                // Read the last i32 offset
                let last_offset_start = length * 4; // (length) * sizeof(i32)
                let last_offset = i32::from_le_bytes(
                    offset_bytes[last_offset_start..last_offset_start + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                last_offset
            };
            let child = reconstruct_array(
                field.data_type(),
                child_length,
                null_counts,
                buf_lengths,
                buf_data,
                node_idx,
                buf_idx,
                data_offset,
            )?;
            vec![child]
        }
        DataType::Struct(fields) => {
            let mut children = Vec::with_capacity(fields.len());
            for field in fields {
                let child = reconstruct_array(
                    field.data_type(),
                    length,
                    null_counts,
                    buf_lengths,
                    buf_data,
                    node_idx,
                    buf_idx,
                    data_offset,
                )?;
                children.push(child);
            }
            children
        }
        DataType::Map(field, _) => {
            // Map has offsets + 1 child (struct with key, value)
            let offsets_buf = &buffers[0];
            let child_length = if length == 0 {
                0
            } else {
                let offset_bytes = &offsets_buf.as_slice();
                let last_offset_start = length * 4;
                let last_offset = i32::from_le_bytes(
                    offset_bytes[last_offset_start..last_offset_start + 4]
                        .try_into()
                        .unwrap(),
                ) as usize;
                last_offset
            };
            let child = reconstruct_array(
                field.data_type(),
                child_length,
                null_counts,
                buf_lengths,
                buf_data,
                node_idx,
                buf_idx,
                data_offset,
            )?;
            vec![child]
        }
        _ => vec![],
    };

    let builder = ArrayData::builder(dt.clone())
        .len(length)
        .null_count(null_count)
        .buffers(buffers)
        .child_data(child_data);

    let builder = if let Some(null_buf) = null_buffer {
        builder.null_bit_buffer(Some(null_buf))
    } else {
        builder
    };

    // SAFETY: We trust the data written by the CSB1 writer.
    unsafe { builder.build_unchecked() }.pipe(Ok)
}

/// Helper trait to allow `Ok(value)` chaining.
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R
    where
        F: FnOnce(Self) -> R,
    {
        f(self)
    }
}
impl<T> Pipe for T {}

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
        })
    }

    /// Writes given record batch in CSB1 format into given writer.
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

        // write header (block_size placeholder + field_count + codec)
        output.write_all(&self.header_bytes)?;

        // Collect all array buffers in DFS order
        // We must first collect ArrayData to keep references alive for buf_data slices
        let array_datas: Vec<ArrayData> = batch.columns().iter().map(|c| c.to_data()).collect();
        let mut null_counts: Vec<u32> = Vec::new();
        let mut buf_lengths: Vec<u32> = Vec::new();
        let mut buf_data: Vec<&[u8]> = Vec::new();

        for data in &array_datas {
            collect_array_buffers(data, &mut null_counts, &mut buf_lengths, &mut buf_data);
        }

        let num_rows = batch.num_rows() as u32;
        let num_nodes = null_counts.len() as u16;
        let num_buffers = buf_lengths.len() as u16;

        // Build the CSB1 payload
        let mut payload = Vec::new();
        payload.write_all(CSB1_MAGIC)?;
        payload.write_all(&num_rows.to_le_bytes())?;
        payload.write_all(&num_nodes.to_le_bytes())?;
        payload.write_all(&num_buffers.to_le_bytes())?;

        // Write null counts
        for &nc in &null_counts {
            payload.write_all(&nc.to_le_bytes())?;
        }

        // Write buffer lengths
        for &bl in &buf_lengths {
            payload.write_all(&bl.to_le_bytes())?;
        }

        // Write concatenated buffer data (no alignment padding)
        for &data in &buf_data {
            payload.write_all(data)?;
        }

        // Compress and write payload
        match &self.codec {
            CompressionCodec::None => {
                output.write_all(&payload)?;
            }
            CompressionCodec::Lz4Frame => {
                let mut encoder = lz4_flex::frame::FrameEncoder::new(output.by_ref());
                encoder.write_all(&payload)?;
                encoder.finish().map_err(|e| {
                    DataFusionError::Execution(format!("lz4 compression error: {e}"))
                })?;
            }
            CompressionCodec::Zstd(level) => {
                let mut encoder = zstd::Encoder::new(output.by_ref(), *level)?;
                encoder.write_all(&payload)?;
                encoder.finish()?;
            }
            CompressionCodec::Snappy => {
                let mut encoder = snap::write::FrameEncoder::new(output.by_ref());
                encoder.write_all(&payload)?;
                encoder.into_inner().map_err(|e| {
                    DataFusionError::Execution(format!("snappy compression error: {e}"))
                })?;
            }
        }

        // fill block_size
        let end_pos = output.stream_position()?;
        let block_size = end_pos - start_pos - 8;
        let max_size = i32::MAX as u64;
        if block_size > max_size {
            return Err(DataFusionError::Execution(format!(
                "Shuffle block size {block_size} exceeds maximum size of {max_size}. \
                Try reducing batch size or increasing compression level"
            )));
        }

        output.seek(SeekFrom::Start(start_pos))?;
        output.write_all(&block_size.to_le_bytes())?;
        output.seek(SeekFrom::Start(end_pos))?;

        timer.stop();

        Ok((end_pos - start_pos) as usize)
    }
}

/// Decompress a CSB1 shuffle block.
/// `bytes` starts after the 8-byte block_size prefix and 8-byte field_count,
/// i.e., it begins with the 4-byte codec identifier.
fn decompress_csb1_payload(bytes: &[u8]) -> Result<Vec<u8>> {
    match &bytes[0..4] {
        b"SNAP" => {
            let mut decoder = snap::read::FrameDecoder::new(&bytes[4..]);
            let mut payload = Vec::new();
            decoder.read_to_end(&mut payload).map_err(|e| {
                DataFusionError::Execution(format!("snappy decompression error: {e}"))
            })?;
            Ok(payload)
        }
        b"LZ4_" => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(&bytes[4..]);
            let mut payload = Vec::new();
            decoder
                .read_to_end(&mut payload)
                .map_err(|e| DataFusionError::Execution(format!("lz4 decompression error: {e}")))?;
            Ok(payload)
        }
        b"ZSTD" => {
            let mut decoder = zstd::Decoder::new(&bytes[4..])?;
            let mut payload = Vec::new();
            decoder.read_to_end(&mut payload).map_err(|e| {
                DataFusionError::Execution(format!("zstd decompression error: {e}"))
            })?;
            Ok(payload)
        }
        b"NONE" => Ok(bytes[4..].to_vec()),
        other => Err(DataFusionError::Execution(format!(
            "Failed to decode batch: invalid compression codec: {other:?}"
        ))),
    }
}

/// Read a CSB1-encoded shuffle block and reconstruct a RecordBatch.
///
/// `bytes` starts after the 8-byte block_size prefix and 8-byte field_count,
/// i.e., it begins with the 4-byte codec identifier.
/// The `schema` must match the schema used when writing.
pub fn read_csb1(bytes: &[u8], schema: &Schema) -> Result<RecordBatch> {
    let payload = decompress_csb1_payload(bytes)?;

    // Parse CSB1 header
    if payload.len() < 12 {
        return Err(DataFusionError::Execution(
            "CSB1 payload too short for header".to_string(),
        ));
    }

    let magic = &payload[0..4];
    if magic != CSB1_MAGIC {
        return Err(DataFusionError::Execution(format!(
            "Invalid CSB1 magic: expected {:?}, got {:?}",
            CSB1_MAGIC, magic
        )));
    }

    let num_rows = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let num_nodes = u16::from_le_bytes(payload[8..10].try_into().unwrap()) as usize;
    let num_buffers = u16::from_le_bytes(payload[10..12].try_into().unwrap()) as usize;

    // Read null_counts and buf_lengths
    let mut offset = 12;
    let null_counts: Vec<u32> = (0..num_nodes)
        .map(|_| {
            let val = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            offset += 4;
            val
        })
        .collect();

    let buf_lengths: Vec<u32> = (0..num_buffers)
        .map(|_| {
            let val = u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            offset += 4;
            val
        })
        .collect();

    // The rest is buffer data
    let buf_data = &payload[offset..];

    // Reconstruct columns
    let mut node_idx = 0;
    let mut buf_idx = 0;
    let mut data_offset = 0;

    let mut columns = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let array_data = reconstruct_array(
            field.data_type(),
            num_rows,
            &null_counts,
            &buf_lengths,
            buf_data,
            &mut node_idx,
            &mut buf_idx,
            &mut data_offset,
        )?;
        columns.push(arrow::array::make_array(array_data));
    }

    RecordBatch::try_new(Arc::new(schema.clone()), columns).map_err(|e| e.into())
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
