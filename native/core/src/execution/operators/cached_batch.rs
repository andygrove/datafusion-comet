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

//! Decoding of Comet's in-memory cache payloads.
//!
//! A cached batch is one encapsulated Arrow IPC RecordBatch message followed by its body, with no
//! Schema message and no end-of-stream marker, written on the JVM by `CachedBatchIpc.serialize`.
//! Compression is applied per buffer, so a projected read decompresses only the buffers of the
//! columns it selects. The schema is not stored: the reader rebuilds it from the cached relation's
//! Spark types, which is what [`CachedBatchLayout`] holds.

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::ipc::reader::read_record_batch;
use arrow::ipc::root_as_message;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use std::collections::HashMap;

use super::scan::schema_from_data_types;

/// The cached relation's schema and the columns a scan reads from it.
#[derive(Debug, Clone)]
pub struct CachedBatchLayout {
    /// Every column of the cached relation, in the order the cache stores them.
    relation_schema: SchemaRef,
    /// For each column the scan outputs, its index in `relation_schema`.
    projection: Vec<usize>,
}

impl CachedBatchLayout {
    pub fn try_new(
        relation_types: Vec<DataType>,
        projection: Vec<usize>,
    ) -> DataFusionResult<Self> {
        if let Some(bad) = projection.iter().find(|&&i| i >= relation_types.len()) {
            return Err(DataFusionError::Internal(format!(
                "Cached batch projection index {bad} is outside the {} cached columns",
                relation_types.len()
            )));
        }
        Ok(Self {
            relation_schema: schema_from_data_types(&relation_types),
            projection,
        })
    }

    /// Decode the projected columns of one cached payload.
    ///
    /// `bytes` is only borrowed: the body is copied into an allocation of its own before decoding,
    /// because a buffer stored uncompressed comes back as a slice of the body, and the JVM reuses
    /// the memory behind `bytes` for the next payload.
    pub fn decode(&self, bytes: &[u8]) -> DataFusionResult<RecordBatch> {
        let (metadata_start, metadata_len): (usize, i32) = match bytes {
            // The continuation marker, then the metadata length.
            [0xff, 0xff, 0xff, 0xff, a, b, c, d, ..] => (8, i32::from_le_bytes([*a, *b, *c, *d])),
            // The pre-0.15 framing, without the continuation marker.
            [a, b, c, d, ..] => (4, i32::from_le_bytes([*a, *b, *c, *d])),
            _ => return Err(corrupt(format!("{} bytes is too short", bytes.len()))),
        };
        let metadata_end = usize::try_from(metadata_len)
            .ok()
            .and_then(|len| metadata_start.checked_add(len))
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| {
                corrupt(format!(
                    "metadata length {metadata_len} runs past the {}-byte payload",
                    bytes.len()
                ))
            })?;

        let message = root_as_message(&bytes[metadata_start..metadata_end])
            .map_err(|e| corrupt(format!("invalid message metadata: {e}")))?;
        let batch = message
            .header_as_record_batch()
            .ok_or_else(|| corrupt("the message is not a RecordBatch".to_string()))?;

        let body_end = usize::try_from(message.bodyLength())
            .ok()
            .and_then(|len| metadata_end.checked_add(len))
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| {
                corrupt(format!(
                    "body length {} runs past the {}-byte payload",
                    message.bodyLength(),
                    bytes.len()
                ))
            })?;
        let body = Buffer::from(&bytes[metadata_end..body_end]);

        let decoded = read_record_batch(
            &body,
            batch,
            SchemaRef::clone(&self.relation_schema),
            &HashMap::new(),
            Some(&self.projection),
            &message.version(),
        )?;
        Ok(decoded)
    }
}

fn corrupt(detail: String) -> DataFusionError {
    DataFusionError::Execution(format!("Corrupt Comet cached batch: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use arrow::ipc::writer::{write_message, DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
    use arrow::ipc::CompressionType;
    use std::sync::Arc;

    /// Encode `batch` the way `CachedBatchIpc.serialize` does: one encapsulated RecordBatch
    /// message and its body, with no Schema message and no end-of-stream marker.
    fn encode(batch: &RecordBatch, compression: Option<CompressionType>) -> Vec<u8> {
        let options = IpcWriteOptions::default()
            .try_with_compression(compression)
            .unwrap();
        let generator = IpcDataGenerator::default();
        let mut tracker = DictionaryTracker::new(false);
        let (_, encoded) = generator
            .encode(batch, &mut tracker, &options, &mut Default::default())
            .unwrap();
        let mut out = Vec::new();
        write_message(&mut out, encoded, &options).unwrap();
        out
    }

    fn relation() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Int64, true),
        ]));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
            Arc::new(StringArray::from(vec![Some("x"), Some("yy"), None])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ];
        RecordBatch::try_new(schema, columns).unwrap()
    }

    fn layout(projection: Vec<usize>) -> CachedBatchLayout {
        CachedBatchLayout::try_new(
            vec![DataType::Int32, DataType::Utf8, DataType::Int64],
            projection,
        )
        .unwrap()
    }

    #[test]
    fn decodes_projected_columns_under_each_codec() {
        let batch = relation();
        for codec in [
            None,
            Some(CompressionType::ZSTD),
            Some(CompressionType::LZ4_FRAME),
        ] {
            let bytes = encode(&batch, codec);
            let decoded = layout(vec![2, 0]).decode(&bytes).unwrap();
            assert_eq!(decoded.num_rows(), 3);
            assert_eq!(decoded.num_columns(), 2);
            assert_eq!(decoded.column(0).as_ref(), batch.column(2).as_ref());
            assert_eq!(decoded.column(1).as_ref(), batch.column(0).as_ref());
        }
    }

    #[test]
    fn decodes_a_column_that_follows_a_variable_width_one() {
        let batch = relation();
        let bytes = encode(&batch, Some(CompressionType::ZSTD));
        let decoded = layout(vec![1, 2]).decode(&bytes).unwrap();
        assert_eq!(decoded.column(0).as_ref(), batch.column(1).as_ref());
        assert_eq!(decoded.column(1).as_ref(), batch.column(2).as_ref());
        assert_eq!(decoded.column(0).null_count(), 1);
    }

    #[test]
    fn rejects_truncated_payloads() {
        let bytes = encode(&relation(), None);
        for len in [0, 3, 7, 20, bytes.len() - 1] {
            assert!(
                layout(vec![0]).decode(&bytes[..len]).is_err(),
                "length {len}"
            );
        }
    }

    #[test]
    fn rejects_projection_outside_the_relation() {
        assert!(CachedBatchLayout::try_new(vec![DataType::Int32], vec![1]).is_err());
    }
}
