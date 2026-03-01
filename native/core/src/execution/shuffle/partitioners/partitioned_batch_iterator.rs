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

/// A helper struct to produce partitioned batches.
/// This struct takes ownership of pre-partitioned sub-batches and provides an iterator
/// over the batches for a specified partition.
pub(super) struct PartitionedBatchesProducer {
    partition_batches: Vec<Vec<RecordBatch>>,
}

impl PartitionedBatchesProducer {
    pub(super) fn new(partition_batches: Vec<Vec<RecordBatch>>) -> Self {
        Self { partition_batches }
    }

    pub(super) fn produce(&mut self, partition_id: usize) -> PartitionedBatchIterator<'_> {
        PartitionedBatchIterator {
            batches: &self.partition_batches[partition_id],
            pos: 0,
        }
    }
}

pub(crate) struct PartitionedBatchIterator<'a> {
    batches: &'a [RecordBatch],
    pos: usize,
}

impl<'a> Iterator for PartitionedBatchIterator<'a> {
    type Item = datafusion::common::Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.batches.len() {
            return None;
        }
        let batch = self.batches[self.pos].clone();
        self.pos += 1;
        Some(Ok(batch))
    }
}
