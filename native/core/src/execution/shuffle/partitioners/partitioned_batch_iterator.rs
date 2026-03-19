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

use arrow::array::RecordBatch;
use arrow::compute::interleave_record_batch;
use datafusion::common::DataFusionError;

/// A helper struct to produce shuffled batches.
/// This struct takes ownership of the buffered batches and partition indices from the
/// ShuffleRepartitioner, and provides an iterator over the batches in the specified partitions.
pub(super) struct PartitionedBatchesProducer {
    buffered_batches: Vec<RecordBatch>,
    partition_indices: Vec<Vec<(usize, usize)>>,
    batch_size: usize,
}

impl PartitionedBatchesProducer {
    pub(super) fn new(
        buffered_batches: Vec<RecordBatch>,
        mut indices: Vec<Vec<(usize, usize)>>,
        batch_size: usize,
    ) -> Self {
        // Sort each partition's indices by (batch_id, row_id) to improve data locality
        // during the interleave/gather step. Arrow's interleave_record_batch has an
        // internal optimization that coalesces adjacent rows from the same source batch
        // into a single extend() call, which is much faster than per-row random access.
        for partition_indices in &mut indices {
            partition_indices.sort_unstable();
        }
        Self {
            partition_indices: indices,
            buffered_batches,
            batch_size,
        }
    }

    pub(super) fn produce(&mut self, partition_id: usize) -> PartitionedBatchIterator<'_> {
        PartitionedBatchIterator::new(
            &self.partition_indices[partition_id],
            &self.buffered_batches,
            self.batch_size,
        )
    }
}

pub(crate) struct PartitionedBatchIterator<'a> {
    record_batches: &'a [RecordBatch],
    batch_size: usize,
    // Indices are already (usize, usize) — no conversion needed
    indices: &'a [(usize, usize)],
    pos: usize,
}

impl<'a> PartitionedBatchIterator<'a> {
    fn new(
        indices: &'a [(usize, usize)],
        buffered_batches: &'a [RecordBatch],
        batch_size: usize,
    ) -> Self {
        Self {
            record_batches: buffered_batches,
            batch_size,
            indices,
            pos: 0,
        }
    }
}

impl Iterator for PartitionedBatchIterator<'_> {
    type Item = datafusion::common::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let indices_end = std::cmp::min(self.pos + self.batch_size, self.indices.len());
        let indices = &self.indices[self.pos..indices_end];
        // interleave_record_batch accepts &[&RecordBatch] — collect refs from slice
        let batch_refs: Vec<&RecordBatch> = self.record_batches.iter().collect();
        match interleave_record_batch(&batch_refs, indices) {
            Ok(batch) => {
                self.pos = indices_end;
                Some(Ok(batch))
            }
            Err(e) => Some(Err(DataFusionError::ArrowError(
                Box::from(e),
                Some(DataFusionError::get_back_trace()),
            ))),
        }
    }
}
