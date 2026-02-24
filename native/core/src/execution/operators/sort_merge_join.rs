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

//! CometSortMergeJoinExec - A Spark-optimized Sort-Merge Join operator.
//!
//! Forked from DataFusion 51.0.0's `SortMergeJoinExec` to allow Comet-specific
//! optimizations while maintaining the proven sort-merge join algorithm.
//!
//! Key differences from DataFusion's SortMergeJoinExec:
//! - **Hardcoded NullEquality::NullEqualsNothing**: Spark always treats NULL keys
//!   as non-matching in join predicates. If `EqualNullSafe` is needed, Spark
//!   rewrites the plan during planning.
//! - **Display name**: Shows as `CometSortMergeJoinExec` in explain plans.
//!
//! Future optimizations planned:
//! - Null-key batch pre-filtering
//! - Buffer reuse for consecutive duplicate keys
//! - ExistenceJoin support
//! - Spark-compatible spill/buffer metrics
//! - Built-in batch coalescing

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::fmt::Formatter;
use std::fs::File;
use std::io::BufReader;
use std::mem::size_of;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{types::UInt64Type, *};
use arrow::compute::{
    self, and, concat_batches, filter_record_batch, is_not_null, is_null, take,
    SortOptions,
};
use arrow::compute::kernels::cmp::eq;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamReader;
use datafusion::common::cast::as_boolean_array;
use datafusion::common::{
    config::SpillCompression, exec_err, internal_err, DataFusionError,
    HashSet, JoinSide, JoinType, NullEquality, Result,
};
use datafusion::execution::disk_manager::RefCountedTempFile;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{
    equivalence::join_equivalence_properties, LexOrdering, OrderingRequirements,
    PhysicalExpr, PhysicalExprRef,
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::expressions::PhysicalSortExpr;
use datafusion::physical_plan::joins::utils::{
    build_join_schema, check_join_is_valid, compare_join_arrays, JoinFilter, JoinOn,
    JoinOnRef,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricsSet,
    SpillMetrics, Time,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, RecordBatchStream, SendableRecordBatchStream,
    SpillManager, Statistics,
};
use futures::{Stream, StreamExt};

// ============================================================================
// Metrics
// ============================================================================

/// Metrics for CometSortMergeJoinExec
#[allow(dead_code)]
struct CometSortMergeJoinMetrics {
    /// Total time for joining probe-side batches to the build-side batches
    join_time: Time,
    /// Number of batches consumed by this operator
    input_batches: Count,
    /// Number of rows consumed by this operator
    input_rows: Count,
    /// Number of batches produced by this operator
    output_batches: Count,
    /// Execution metrics
    baseline_metrics: BaselineMetrics,
    /// Peak memory used for buffered data.
    peak_mem_used: Gauge,
    /// Metrics related to spilling
    spill_metrics: SpillMetrics,
}

impl CometSortMergeJoinMetrics {
    #[allow(dead_code)]
    fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        let join_time = MetricBuilder::new(metrics).subset_time("join_time", partition);
        let input_batches =
            MetricBuilder::new(metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(metrics).counter("input_rows", partition);
        let output_batches =
            MetricBuilder::new(metrics).counter("output_batches", partition);
        let peak_mem_used = MetricBuilder::new(metrics).gauge("peak_mem_used", partition);
        let spill_metrics = SpillMetrics::new(metrics, partition);
        let baseline_metrics = BaselineMetrics::new(metrics, partition);

        Self {
            join_time,
            input_batches,
            input_rows,
            output_batches,
            baseline_metrics,
            peak_mem_used,
            spill_metrics,
        }
    }

    fn join_time(&self) -> Time {
        self.join_time.clone()
    }

    fn baseline_metrics(&self) -> BaselineMetrics {
        self.baseline_metrics.clone()
    }

    fn input_batches(&self) -> Count {
        self.input_batches.clone()
    }

    fn input_rows(&self) -> Count {
        self.input_rows.clone()
    }

    fn output_batches(&self) -> Count {
        self.output_batches.clone()
    }

    fn peak_mem_used(&self) -> Gauge {
        self.peak_mem_used.clone()
    }

    fn spill_metrics(&self) -> SpillMetrics {
        self.spill_metrics.clone()
    }
}

// ============================================================================
// CometSortMergeJoinExec (execution plan node)
// ============================================================================

/// Comet's Sort-Merge Join execution plan, forked from DataFusion 51.0.0.
///
/// This operator always uses `NullEquality::NullEqualsNothing` since Spark
/// never matches NULL keys in equi-joins (EqualNullSafe is rewritten by Spark
/// during planning).
#[derive(Debug, Clone)]
pub struct CometSortMergeJoinExec {
    /// Left sorted joining execution plan
    pub left: Arc<dyn ExecutionPlan>,
    /// Right sorting joining execution plan
    pub right: Arc<dyn ExecutionPlan>,
    /// Set of common columns used to join on
    pub on: JoinOn,
    /// Filters which are applied while finding matching rows
    pub filter: Option<JoinFilter>,
    /// How the join is performed
    pub join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// The left SortExpr
    left_sort_exprs: LexOrdering,
    /// The right SortExpr
    right_sort_exprs: LexOrdering,
    /// Sort options of join columns used in sorting left and right execution plans
    pub sort_options: Vec<SortOptions>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
}

impl CometSortMergeJoinExec {
    /// Tries to create a new [CometSortMergeJoinExec].
    /// The inputs are sorted using `sort_options` applied to the columns in `on`.
    ///
    /// Unlike DataFusion's SortMergeJoinExec, this does not take a `null_equality`
    /// parameter — Spark always uses NullEqualsNothing for join keys.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: JoinOn,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        sort_options: Vec<SortOptions>,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();

        check_join_is_valid(&left_schema, &right_schema, &on)?;
        if sort_options.len() != on.len() {
            return internal_err!(
                "Expected number of sort options: {}, actual: {}",
                on.len(),
                sort_options.len()
            );
        }

        let (left_sort_exprs, right_sort_exprs): (Vec<_>, Vec<_>) = on
            .iter()
            .zip(sort_options.iter())
            .map(|((l, r), sort_op)| {
                let left = PhysicalSortExpr {
                    expr: Arc::clone(l),
                    options: *sort_op,
                };
                let right = PhysicalSortExpr {
                    expr: Arc::clone(r),
                    options: *sort_op,
                };
                (left, right)
            })
            .unzip();
        let Some(left_sort_exprs) = LexOrdering::new(left_sort_exprs) else {
            return internal_err!(
                "CometSortMergeJoinExec requires valid sort expressions for its left side"
            );
        };
        let Some(right_sort_exprs) = LexOrdering::new(right_sort_exprs) else {
            return internal_err!(
                "CometSortMergeJoinExec requires valid sort expressions for its right side"
            );
        };

        let schema =
            Arc::new(build_join_schema(&left_schema, &right_schema, &join_type).0);
        let cache =
            Self::compute_properties(&left, &right, Arc::clone(&schema), join_type, &on)?;
        Ok(Self {
            left,
            right,
            on,
            filter,
            join_type,
            schema,
            metrics: ExecutionPlanMetricsSet::new(),
            left_sort_exprs,
            right_sort_exprs,
            sort_options,
            cache,
        })
    }

    /// Get probe side (e.g streaming side) information for this sort merge join.
    pub fn probe_side(join_type: &JoinType) -> JoinSide {
        match join_type {
            JoinType::Right
            | JoinType::RightSemi
            | JoinType::RightAnti
            | JoinType::RightMark => JoinSide::Right,
            JoinType::Inner
            | JoinType::Left
            | JoinType::Full
            | JoinType::LeftAnti
            | JoinType::LeftSemi
            | JoinType::LeftMark => JoinSide::Left,
        }
    }

    /// Calculate order preservation flags for this sort merge join.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        match join_type {
            JoinType::Inner => vec![true, false],
            JoinType::Left
            | JoinType::LeftSemi
            | JoinType::LeftAnti
            | JoinType::LeftMark => vec![true, false],
            JoinType::Right
            | JoinType::RightSemi
            | JoinType::RightAnti
            | JoinType::RightMark => {
                vec![false, true]
            }
            _ => vec![false, false],
        }
    }

    /// Set of common columns used to join on
    pub fn on(&self) -> &[(PhysicalExprRef, PhysicalExprRef)] {
        &self.on
    }

    /// Ref to right execution plan
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Join type
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Ref to left execution plan
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// Ref to join filter
    pub fn filter(&self) -> &Option<JoinFilter> {
        &self.filter
    }

    /// Ref to sort options
    pub fn sort_options(&self) -> &[SortOptions] {
        &self.sort_options
    }

    /// This function creates the cache object that stores the plan properties.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        join_on: JoinOnRef,
    ) -> Result<PlanProperties> {
        let eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            schema,
            &Self::maintains_input_order(join_type),
            Some(Self::probe_side(&join_type)),
            join_on,
        )?;

        // Output partitioning follows the streamed (probe) side.
        let output_partitioning = match join_type {
            JoinType::Right | JoinType::RightSemi | JoinType::RightAnti
            | JoinType::RightMark => right.output_partitioning().clone(),
            _ => left.output_partitioning().clone(),
        };

        // Compute boundedness from children: bounded iff all children are bounded.
        let boundedness =
            if [left, right].iter().all(|c| {
                matches!(c.properties().boundedness, Boundedness::Bounded)
            }) {
                Boundedness::Bounded
            } else {
                Boundedness::Unbounded {
                    requires_infinite_memory: true,
                }
            };

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Incremental,
            boundedness,
        ))
    }
}

impl DisplayAs for CometSortMergeJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1}, {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(
                    f,
                    "CometSortMergeJoin: join_type={:?}, on=[{}]{}",
                    self.join_type,
                    on,
                    self.filter.as_ref().map_or_else(
                        || "".to_string(),
                        |f| format!(", filter={}", f.expression())
                    ),
                )
            }
            DisplayFormatType::TreeRender => {
                let on = self
                    .on
                    .iter()
                    .map(|(c1, c2)| format!("({c1} = {c2})"))
                    .collect::<Vec<String>>()
                    .join(", ");

                if self.join_type() != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)?;
                }
                writeln!(f, "on={on}")?;

                Ok(())
            }
        }
    }
}

impl ExecutionPlan for CometSortMergeJoinExec {
    fn name(&self) -> &'static str {
        "CometSortMergeJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        let (left_expr, right_expr) = self
            .on
            .iter()
            .map(|(l, r)| (Arc::clone(l), Arc::clone(r)))
            .unzip();
        vec![
            Distribution::HashPartitioned(left_expr),
            Distribution::HashPartitioned(right_expr),
        ]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![
            Some(OrderingRequirements::from(self.left_sort_exprs.clone())),
            Some(OrderingRequirements::from(self.right_sort_exprs.clone())),
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match &children[..] {
            [left, right] => Ok(Arc::new(CometSortMergeJoinExec::try_new(
                Arc::clone(left),
                Arc::clone(right),
                self.on.clone(),
                self.filter.clone(),
                self.join_type,
                self.sort_options.clone(),
            )?)),
            _ => internal_err!("CometSortMergeJoin wrong number of children"),
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left_partitions = self.left.output_partitioning().partition_count();
        let right_partitions = self.right.output_partitioning().partition_count();
        if left_partitions != right_partitions {
            return internal_err!(
                "Invalid CometSortMergeJoinExec, partition count mismatch {left_partitions}!={right_partitions},\
                 consider using RepartitionExec"
            );
        }
        let (on_left, on_right) = self.on.iter().cloned().unzip();
        let (streamed, buffered, on_streamed, on_buffered) =
            if CometSortMergeJoinExec::probe_side(&self.join_type) == JoinSide::Left {
                (
                    Arc::clone(&self.left),
                    Arc::clone(&self.right),
                    on_left,
                    on_right,
                )
            } else {
                (
                    Arc::clone(&self.right),
                    Arc::clone(&self.left),
                    on_right,
                    on_left,
                )
            };

        // execute children plans
        let streamed = streamed.execute(partition, Arc::clone(&context))?;
        let buffered = buffered.execute(partition, Arc::clone(&context))?;

        // create output buffer
        let batch_size = context.session_config().batch_size();

        // create memory reservation
        let reservation = MemoryConsumer::new(format!("CometSMJStream[{partition}]"))
            .register(context.memory_pool());

        // create join stream
        Ok(Box::pin(CometSortMergeJoinStream::try_new(
            context.session_config().spill_compression(),
            Arc::clone(&self.schema),
            self.sort_options.clone(),
            streamed,
            buffered,
            on_streamed,
            on_buffered,
            self.filter.clone(),
            self.join_type,
            batch_size,
            CometSortMergeJoinMetrics::new(partition, &self.metrics),
            reservation,
            context.runtime_env(),
        )?))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // Join statistics estimation is complex and not critical for Comet.
        // Return unknown for now; can be refined later.
        Ok(Statistics::new_unknown(&self.schema()))
    }
}

// ============================================================================
// Stream types and state machines
// ============================================================================

/// State of SMJ stream
#[derive(Debug, PartialEq, Eq)]
enum CometSortMergeJoinState {
    Init,
    Polling,
    JoinOutput,
    Exhausted,
}

/// State of streamed data stream
#[derive(Debug, PartialEq, Eq)]
enum StreamedState {
    Init,
    Polling,
    Ready,
    Exhausted,
}

/// State of buffered data stream
#[derive(Debug, PartialEq, Eq)]
enum BufferedState {
    Init,
    PollingFirst,
    PollingRest,
    Ready,
    Exhausted,
}

/// Represents a chunk of joined data from streamed and buffered side
struct StreamedJoinedChunk {
    buffered_batch_idx: Option<usize>,
    streamed_indices: UInt64Builder,
    buffered_indices: UInt64Builder,
}

/// Represents a record batch from streamed input.
struct StreamedBatch {
    pub batch: RecordBatch,
    pub idx: usize,
    pub join_arrays: Vec<ArrayRef>,
    pub output_indices: Vec<StreamedJoinedChunk>,
    pub buffered_batch_idx: Option<usize>,
    pub join_filter_matched_idxs: HashSet<u64>,
}

impl StreamedBatch {
    fn new(batch: RecordBatch, on_column: &[Arc<dyn PhysicalExpr>]) -> Self {
        let join_arrays = join_arrays(&batch, on_column);
        StreamedBatch {
            batch,
            idx: 0,
            join_arrays,
            output_indices: vec![],
            buffered_batch_idx: None,
            join_filter_matched_idxs: HashSet::new(),
        }
    }

    fn new_empty(schema: SchemaRef) -> Self {
        StreamedBatch {
            batch: RecordBatch::new_empty(schema),
            idx: 0,
            join_arrays: vec![],
            output_indices: vec![],
            buffered_batch_idx: None,
            join_filter_matched_idxs: HashSet::new(),
        }
    }

    fn append_output_pair(
        &mut self,
        buffered_batch_idx: Option<usize>,
        buffered_idx: Option<usize>,
    ) {
        if self.output_indices.is_empty() || self.buffered_batch_idx != buffered_batch_idx
        {
            self.output_indices.push(StreamedJoinedChunk {
                buffered_batch_idx,
                streamed_indices: UInt64Builder::with_capacity(1),
                buffered_indices: UInt64Builder::with_capacity(1),
            });
            self.buffered_batch_idx = buffered_batch_idx;
        };
        let current_chunk = self.output_indices.last_mut().unwrap();

        current_chunk.streamed_indices.append_value(self.idx as u64);
        if let Some(idx) = buffered_idx {
            current_chunk.buffered_indices.append_value(idx as u64);
        } else {
            current_chunk.buffered_indices.append_null();
        }
    }
}

/// A buffered batch that contains contiguous rows with same join key
#[derive(Debug)]
struct BufferedBatch {
    pub batch: BufferedBatchState,
    pub range: Range<usize>,
    pub join_arrays: Vec<ArrayRef>,
    pub null_joined: Vec<usize>,
    pub size_estimation: usize,
    pub join_filter_not_matched_map: HashMap<u64, bool>,
    pub num_rows: usize,
    /// Cache for deserialized spilled batches to avoid redundant disk I/O
    pub spill_cache: Option<RecordBatch>,
}

impl BufferedBatch {
    fn new(
        batch: RecordBatch,
        range: Range<usize>,
        on_column: &[PhysicalExprRef],
    ) -> Self {
        let join_arrays = join_arrays(&batch, on_column);

        let size_estimation = batch.get_array_memory_size()
            + join_arrays
                .iter()
                .map(|arr| arr.get_array_memory_size())
                .sum::<usize>()
            + batch.num_rows().next_power_of_two() * size_of::<usize>()
            + size_of::<Range<usize>>()
            + size_of::<usize>();

        let num_rows = batch.num_rows();
        BufferedBatch {
            batch: BufferedBatchState::InMemory(batch),
            range,
            join_arrays,
            null_joined: vec![],
            size_estimation,
            join_filter_not_matched_map: HashMap::new(),
            num_rows,
            spill_cache: None,
        }
    }
}

#[derive(Debug)]
enum BufferedBatchState {
    InMemory(RecordBatch),
    Spilled(RefCountedTempFile),
}

// ============================================================================
// CometSortMergeJoinStream
// ============================================================================

/// Sort-Merge join stream that consumes streamed and buffered data streams
/// and produces joined output.
struct CometSortMergeJoinStream {
    // -- Properties (constant throughout execution) --
    pub schema: SchemaRef,
    pub sort_options: Vec<SortOptions>,
    pub filter: Option<JoinFilter>,
    pub join_type: JoinType,
    pub batch_size: usize,

    // -- Streamed side --
    pub streamed_schema: SchemaRef,
    pub streamed: SendableRecordBatchStream,
    pub streamed_batch: StreamedBatch,
    pub streamed_joined: bool,
    pub streamed_state: StreamedState,
    pub on_streamed: Vec<PhysicalExprRef>,

    // -- Buffered side --
    pub buffered_schema: SchemaRef,
    pub buffered: SendableRecordBatchStream,
    pub buffered_data: BufferedData,
    pub buffered_joined: bool,
    pub buffered_state: BufferedState,
    pub on_buffered: Vec<PhysicalExprRef>,

    // -- Merge join state --
    pub state: CometSortMergeJoinState,
    pub staging_output_record_batches: JoinedRecordBatches,
    pub output: RecordBatch,
    pub output_size: usize,
    pub current_ordering: Ordering,
    pub spill_manager: SpillManager,

    // -- Execution resources --
    pub join_metrics: CometSortMergeJoinMetrics,
    pub reservation: MemoryReservation,
    pub runtime_env: Arc<RuntimeEnv>,
    pub streamed_batch_counter: AtomicUsize,
}

/// Joined batches with attached join filter information
struct JoinedRecordBatches {
    pub batches: Vec<RecordBatch>,
    pub filter_mask: BooleanBuilder,
    pub row_indices: UInt64Builder,
    pub batch_ids: Vec<usize>,
}

impl JoinedRecordBatches {
    fn clear(&mut self) {
        self.batches.clear();
        self.batch_ids.clear();
        self.filter_mask = BooleanBuilder::new();
        self.row_indices = UInt64Builder::new();
    }
}

impl RecordBatchStream for CometSortMergeJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// True if next index refers to either another batch id, another row index, or end of indices
#[inline(always)]
fn last_index_for_row(
    row_index: usize,
    indices: &UInt64Array,
    batch_ids: &[usize],
    indices_len: usize,
) -> bool {
    row_index == indices_len - 1
        || batch_ids[row_index] != batch_ids[row_index + 1]
        || indices.value(row_index) != indices.value(row_index + 1)
}

/// Returns a corrected boolean bitmask for the given join type
fn get_corrected_filter_mask(
    join_type: JoinType,
    row_indices: &UInt64Array,
    batch_ids: &[usize],
    filter_mask: &BooleanArray,
    expected_size: usize,
) -> Option<BooleanArray> {
    let row_indices_length = row_indices.len();
    let mut corrected_mask: BooleanBuilder =
        BooleanBuilder::with_capacity(row_indices_length);
    let mut seen_true = false;

    match join_type {
        JoinType::Left | JoinType::Right => {
            for i in 0..row_indices_length {
                let last_index =
                    last_index_for_row(i, row_indices, batch_ids, row_indices_length);
                if filter_mask.value(i) {
                    seen_true = true;
                    corrected_mask.append_value(true);
                } else if seen_true || !filter_mask.value(i) && !last_index {
                    corrected_mask.append_null();
                } else {
                    corrected_mask.append_value(false);
                }

                if last_index {
                    seen_true = false;
                }
            }

            corrected_mask.append_n(expected_size - corrected_mask.len(), false);
            Some(corrected_mask.finish())
        }
        JoinType::LeftMark | JoinType::RightMark => {
            for i in 0..row_indices_length {
                let last_index =
                    last_index_for_row(i, row_indices, batch_ids, row_indices_length);
                if filter_mask.value(i) && !seen_true {
                    seen_true = true;
                    corrected_mask.append_value(true);
                } else if seen_true || !filter_mask.value(i) && !last_index {
                    corrected_mask.append_null();
                } else {
                    corrected_mask.append_value(false);
                }

                if last_index {
                    seen_true = false;
                }
            }

            corrected_mask.append_n(expected_size - corrected_mask.len(), false);
            Some(corrected_mask.finish())
        }
        JoinType::LeftSemi | JoinType::RightSemi => {
            for i in 0..row_indices_length {
                let last_index =
                    last_index_for_row(i, row_indices, batch_ids, row_indices_length);
                if filter_mask.value(i) && !seen_true {
                    seen_true = true;
                    corrected_mask.append_value(true);
                } else {
                    corrected_mask.append_null();
                }

                if last_index {
                    seen_true = false;
                }
            }

            Some(corrected_mask.finish())
        }
        JoinType::LeftAnti | JoinType::RightAnti => {
            for i in 0..row_indices_length {
                let last_index =
                    last_index_for_row(i, row_indices, batch_ids, row_indices_length);

                if filter_mask.value(i) {
                    seen_true = true;
                }

                if last_index {
                    if !seen_true {
                        corrected_mask.append_value(true);
                    } else {
                        corrected_mask.append_null();
                    }

                    seen_true = false;
                } else {
                    corrected_mask.append_null();
                }
            }
            corrected_mask.append_n(expected_size - corrected_mask.len(), true);
            Some(corrected_mask.finish())
        }
        JoinType::Full => {
            let mut mask: Vec<Option<bool>> = vec![Some(true); row_indices_length];
            let mut last_true_idx = 0;
            let mut first_row_idx = 0;
            let mut seen_false = false;

            for i in 0..row_indices_length {
                let last_index =
                    last_index_for_row(i, row_indices, batch_ids, row_indices_length);
                let val = filter_mask.value(i);
                let is_null = filter_mask.is_null(i);

                if val {
                    if !seen_true {
                        last_true_idx = i;
                    }
                    seen_true = true;
                }

                if is_null || val {
                    mask[i] = Some(true);
                } else if !is_null && !val && (seen_true || seen_false) {
                    mask[i] = None;
                } else {
                    mask[i] = Some(false);
                }

                if !is_null && !val {
                    seen_false = true;
                }

                if last_index {
                    if seen_true {
                        #[allow(clippy::needless_range_loop)]
                        for j in first_row_idx..last_true_idx {
                            mask[j] = None;
                        }
                    }

                    seen_true = false;
                    seen_false = false;
                    last_true_idx = 0;
                    first_row_idx = i + 1;
                }
            }

            Some(BooleanArray::from(mask))
        }
        _ => None,
    }
}

impl Stream for CometSortMergeJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let join_time = self.join_metrics.join_time().clone();
        let _timer = join_time.timer();
        loop {
            match &self.state {
                CometSortMergeJoinState::Init => {
                    let streamed_exhausted =
                        self.streamed_state == StreamedState::Exhausted;
                    let buffered_exhausted =
                        self.buffered_state == BufferedState::Exhausted;
                    self.state = if streamed_exhausted && buffered_exhausted {
                        CometSortMergeJoinState::Exhausted
                    } else {
                        match self.current_ordering {
                            Ordering::Less | Ordering::Equal => {
                                if !streamed_exhausted {
                                    if self.filter.is_some()
                                        && matches!(
                                            self.join_type,
                                            JoinType::Left
                                                | JoinType::LeftSemi
                                                | JoinType::LeftMark
                                                | JoinType::Right
                                                | JoinType::RightSemi
                                                | JoinType::RightMark
                                                | JoinType::LeftAnti
                                                | JoinType::RightAnti
                                                | JoinType::Full
                                        )
                                    {
                                        self.freeze_all()?;

                                        if !self
                                            .staging_output_record_batches
                                            .batches
                                            .is_empty()
                                        {
                                            let out_filtered_batch =
                                                self.filter_joined_batch()?;

                                            self.output = concat_batches(
                                                &self.schema(),
                                                [&self.output, &out_filtered_batch],
                                            )?;

                                            if self.output.num_rows() >= self.batch_size {
                                                let record_batch = std::mem::replace(
                                                    &mut self.output,
                                                    RecordBatch::new_empty(
                                                        out_filtered_batch.schema(),
                                                    ),
                                                );
                                                return Poll::Ready(Some(Ok(
                                                    record_batch,
                                                )));
                                            }
                                        }
                                    }

                                    self.streamed_joined = false;
                                    self.streamed_state = StreamedState::Init;
                                }
                            }
                            Ordering::Greater => {
                                if !buffered_exhausted {
                                    self.buffered_joined = false;
                                    self.buffered_state = BufferedState::Init;
                                }
                            }
                        }
                        CometSortMergeJoinState::Polling
                    };
                }
                CometSortMergeJoinState::Polling => {
                    if ![StreamedState::Exhausted, StreamedState::Ready]
                        .contains(&self.streamed_state)
                    {
                        match self.poll_streamed_row(cx)? {
                            Poll::Ready(_) => {}
                            Poll::Pending => return Poll::Pending,
                        }
                    }

                    if ![BufferedState::Exhausted, BufferedState::Ready]
                        .contains(&self.buffered_state)
                    {
                        match self.poll_buffered_batches(cx)? {
                            Poll::Ready(_) => {}
                            Poll::Pending => return Poll::Pending,
                        }
                    }
                    let streamed_exhausted =
                        self.streamed_state == StreamedState::Exhausted;
                    let buffered_exhausted =
                        self.buffered_state == BufferedState::Exhausted;
                    if streamed_exhausted && buffered_exhausted {
                        self.state = CometSortMergeJoinState::Exhausted;
                        continue;
                    }
                    self.current_ordering = self.compare_streamed_buffered()?;
                    self.state = CometSortMergeJoinState::JoinOutput;
                }
                CometSortMergeJoinState::JoinOutput => {
                    self.join_partial()?;

                    if self.output_size < self.batch_size {
                        if self.buffered_data.scanning_finished() {
                            self.buffered_data.scanning_reset();
                            self.state = CometSortMergeJoinState::Init;
                        }
                    } else {
                        self.freeze_all()?;
                        if !self.staging_output_record_batches.batches.is_empty() {
                            let record_batch = self.output_record_batch_and_reset()?;
                            if self.filter.is_some()
                                && matches!(
                                    self.join_type,
                                    JoinType::Left
                                        | JoinType::LeftSemi
                                        | JoinType::Right
                                        | JoinType::RightSemi
                                        | JoinType::LeftAnti
                                        | JoinType::RightAnti
                                        | JoinType::LeftMark
                                        | JoinType::RightMark
                                        | JoinType::Full
                                )
                            {
                                continue;
                            }

                            return Poll::Ready(Some(Ok(record_batch)));
                        }
                        return Poll::Pending;
                    }
                }
                CometSortMergeJoinState::Exhausted => {
                    self.freeze_all()?;

                    if !self.staging_output_record_batches.batches.is_empty() {
                        if self.filter.is_some()
                            && matches!(
                                self.join_type,
                                JoinType::Left
                                    | JoinType::LeftSemi
                                    | JoinType::Right
                                    | JoinType::RightSemi
                                    | JoinType::LeftAnti
                                    | JoinType::RightAnti
                                    | JoinType::Full
                                    | JoinType::LeftMark
                                    | JoinType::RightMark
                            )
                        {
                            // Coalesce the final filtered batch into self.output
                            // instead of emitting directly. This avoids tiny final
                            // batches and eliminates the need for an external
                            // CoalesceBatchesExec wrapper.
                            let filtered_batch = self.filter_joined_batch()?;
                            if filtered_batch.num_rows() > 0 {
                                self.output = concat_batches(
                                    &self.schema(),
                                    [&self.output, &filtered_batch],
                                )?;
                            }
                            // Fall through to emit self.output below
                        } else {
                            let record_batch = self.output_record_batch_and_reset()?;
                            return Poll::Ready(Some(Ok(record_batch)));
                        }
                    }

                    if self.output.num_rows() > 0 {
                        let schema = self.output.schema();
                        let record_batch = std::mem::replace(
                            &mut self.output,
                            RecordBatch::new_empty(schema),
                        );
                        return Poll::Ready(Some(Ok(record_batch)));
                    } else {
                        return Poll::Ready(None);
                    }
                }
            }
        }
    }
}

impl CometSortMergeJoinStream {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        spill_compression: SpillCompression,
        schema: SchemaRef,
        sort_options: Vec<SortOptions>,
        streamed: SendableRecordBatchStream,
        buffered: SendableRecordBatchStream,
        on_streamed: Vec<Arc<dyn PhysicalExpr>>,
        on_buffered: Vec<Arc<dyn PhysicalExpr>>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        batch_size: usize,
        join_metrics: CometSortMergeJoinMetrics,
        reservation: MemoryReservation,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<Self> {
        let streamed_schema = streamed.schema();
        let buffered_schema = buffered.schema();
        let spill_manager = SpillManager::new(
            Arc::clone(&runtime_env),
            join_metrics.spill_metrics().clone(),
            Arc::clone(&buffered_schema),
        )
        .with_compression_type(spill_compression);
        Ok(Self {
            state: CometSortMergeJoinState::Init,
            sort_options,
            schema: Arc::clone(&schema),
            streamed_schema: Arc::clone(&streamed_schema),
            buffered_schema,
            streamed,
            buffered,
            streamed_batch: StreamedBatch::new_empty(streamed_schema),
            buffered_data: BufferedData::default(),
            streamed_joined: false,
            buffered_joined: false,
            streamed_state: StreamedState::Init,
            buffered_state: BufferedState::Init,
            current_ordering: Ordering::Equal,
            on_streamed,
            on_buffered,
            filter,
            staging_output_record_batches: JoinedRecordBatches {
                batches: vec![],
                filter_mask: BooleanBuilder::new(),
                row_indices: UInt64Builder::new(),
                batch_ids: vec![],
            },
            output: RecordBatch::new_empty(schema),
            output_size: 0,
            batch_size,
            join_type,
            join_metrics,
            reservation,
            runtime_env,
            spill_manager,
            streamed_batch_counter: AtomicUsize::new(0),
        })
    }

    /// Poll next streamed row
    fn poll_streamed_row(&mut self, cx: &mut Context) -> Poll<Option<Result<()>>> {
        loop {
            match &self.streamed_state {
                StreamedState::Init => {
                    if self.streamed_batch.idx + 1 < self.streamed_batch.batch.num_rows()
                    {
                        self.streamed_batch.idx += 1;
                        self.streamed_state = StreamedState::Ready;
                        return Poll::Ready(Some(Ok(())));
                    } else {
                        self.streamed_state = StreamedState::Polling;
                    }
                }
                StreamedState::Polling => match self.streamed.poll_next_unpin(cx)? {
                    Poll::Pending => {
                        return Poll::Pending;
                    }
                    Poll::Ready(None) => {
                        self.streamed_state = StreamedState::Exhausted;
                    }
                    Poll::Ready(Some(batch)) => {
                        if batch.num_rows() > 0 {
                            self.freeze_streamed()?;
                            self.join_metrics.input_batches().add(1);
                            self.join_metrics.input_rows().add(batch.num_rows());
                            self.streamed_batch =
                                StreamedBatch::new(batch, &self.on_streamed);
                            self.streamed_batch_counter
                                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            self.streamed_state = StreamedState::Ready;
                        }
                    }
                },
                StreamedState::Ready => {
                    return Poll::Ready(Some(Ok(())));
                }
                StreamedState::Exhausted => {
                    return Poll::Ready(None);
                }
            }
        }
    }

    fn free_reservation(&mut self, mut buffered_batch: BufferedBatch) -> Result<()> {
        buffered_batch.spill_cache = None;
        if let BufferedBatchState::InMemory(_) = buffered_batch.batch {
            self.reservation
                .try_shrink(buffered_batch.size_estimation)?;
        }
        Ok(())
    }

    fn allocate_reservation(&mut self, mut buffered_batch: BufferedBatch) -> Result<()> {
        match self.reservation.try_grow(buffered_batch.size_estimation) {
            Ok(_) => {
                self.join_metrics
                    .peak_mem_used()
                    .set_max(self.reservation.size());
                Ok(())
            }
            Err(_) if self.runtime_env.disk_manager.tmp_files_enabled() => {
                match buffered_batch.batch {
                    BufferedBatchState::InMemory(batch) => {
                        let spill_file = self
                            .spill_manager
                            .spill_record_batch_and_finish(
                                &[batch],
                                "comet_sort_merge_join_buffered_spill",
                            )?
                            .unwrap();

                        buffered_batch.batch = BufferedBatchState::Spilled(spill_file);
                        Ok(())
                    }
                    _ => internal_err!("Buffered batch has empty body"),
                }
            }
            Err(e) => exec_err!("{}. Disk spilling disabled.", e.message()),
        }?;

        self.buffered_data.batches.push_back(buffered_batch);
        Ok(())
    }

    /// Poll next buffered batches
    fn poll_buffered_batches(&mut self, cx: &mut Context) -> Poll<Option<Result<()>>> {
        loop {
            match &self.buffered_state {
                BufferedState::Init => {
                    while !self.buffered_data.batches.is_empty() {
                        let head_batch = self.buffered_data.head_batch();
                        if head_batch.range.end == head_batch.num_rows {
                            self.freeze_dequeuing_buffered()?;
                            if let Some(mut buffered_batch) =
                                self.buffered_data.batches.pop_front()
                            {
                                self.produce_buffered_not_matched(&mut buffered_batch)?;
                                self.free_reservation(buffered_batch)?;
                            }
                        } else {
                            break;
                        }
                    }
                    if self.buffered_data.batches.is_empty() {
                        self.buffered_state = BufferedState::PollingFirst;
                    } else {
                        let tail_batch = self.buffered_data.tail_batch_mut();
                        tail_batch.range.start = tail_batch.range.end;
                        tail_batch.range.end += 1;
                        self.buffered_state = BufferedState::PollingRest;
                    }
                }
                BufferedState::PollingFirst => match self.buffered.poll_next_unpin(cx)? {
                    Poll::Pending => {
                        return Poll::Pending;
                    }
                    Poll::Ready(None) => {
                        self.buffered_state = BufferedState::Exhausted;
                        return Poll::Ready(None);
                    }
                    Poll::Ready(Some(batch)) => {
                        self.join_metrics.input_batches().add(1);
                        self.join_metrics.input_rows().add(batch.num_rows());

                        if batch.num_rows() > 0 {
                            let buffered_batch =
                                BufferedBatch::new(batch, 0..1, &self.on_buffered);

                            self.allocate_reservation(buffered_batch)?;
                            self.buffered_state = BufferedState::PollingRest;
                        }
                    }
                },
                BufferedState::PollingRest => {
                    if self.buffered_data.tail_batch().range.end
                        < self.buffered_data.tail_batch().num_rows
                    {
                        {
                            let head_arrays =
                                &self.buffered_data.head_batch().join_arrays;
                            let head_start =
                                self.buffered_data.head_batch().range.start;
                            let tail = self.buffered_data.tail_batch();
                            let matching = find_key_group_boundary(
                                head_arrays,
                                head_start,
                                &tail.join_arrays,
                                tail.range.end,
                                tail.num_rows,
                            )?;
                            self.buffered_data.tail_batch_mut().range.end +=
                                matching;

                            if self.buffered_data.tail_batch().range.end
                                < self.buffered_data.tail_batch().num_rows
                            {
                                self.buffered_state = BufferedState::Ready;
                                return Poll::Ready(Some(Ok(())));
                            }
                        }
                    } else {
                        match self.buffered.poll_next_unpin(cx)? {
                            Poll::Pending => {
                                return Poll::Pending;
                            }
                            Poll::Ready(None) => {
                                self.buffered_state = BufferedState::Ready;
                            }
                            Poll::Ready(Some(batch)) => {
                                self.join_metrics.input_batches().add(1);
                                self.join_metrics.input_rows().add(batch.num_rows());
                                if batch.num_rows() > 0 {
                                    let buffered_batch = BufferedBatch::new(
                                        batch,
                                        0..0,
                                        &self.on_buffered,
                                    );
                                    self.allocate_reservation(buffered_batch)?;
                                }
                            }
                        }
                    }
                }
                BufferedState::Ready => {
                    return Poll::Ready(Some(Ok(())));
                }
                BufferedState::Exhausted => {
                    return Poll::Ready(None);
                }
            }
        }
    }

    /// Get comparison result of streamed row and buffered batches.
    /// Always uses NullEquality::NullEqualsNothing (Spark semantics).
    fn compare_streamed_buffered(&self) -> Result<Ordering> {
        if self.streamed_state == StreamedState::Exhausted {
            return Ok(Ordering::Greater);
        }
        if !self.buffered_data.has_buffered_rows() {
            return Ok(Ordering::Less);
        }

        compare_join_arrays(
            &self.streamed_batch.join_arrays,
            self.streamed_batch.idx,
            &self.buffered_data.head_batch().join_arrays,
            self.buffered_data.head_batch().range.start,
            &self.sort_options,
            NullEquality::NullEqualsNothing,
        )
    }

    /// Produce join and fill output buffer until reaching target batch size
    fn join_partial(&mut self) -> Result<()> {
        let mut join_streamed = false;
        let mut join_buffered = false;
        let mut mark_row_as_match = false;

        match self.current_ordering {
            Ordering::Less => {
                if matches!(
                    self.join_type,
                    JoinType::Left
                        | JoinType::Right
                        | JoinType::Full
                        | JoinType::LeftAnti
                        | JoinType::RightAnti
                        | JoinType::LeftMark
                        | JoinType::RightMark
                ) {
                    join_streamed = !self.streamed_joined;
                }
            }
            Ordering::Equal => {
                if matches!(
                    self.join_type,
                    JoinType::LeftSemi
                        | JoinType::LeftMark
                        | JoinType::RightSemi
                        | JoinType::RightMark
                ) {
                    mark_row_as_match = matches!(
                        self.join_type,
                        JoinType::LeftMark | JoinType::RightMark
                    );
                    if self.filter.is_some() {
                        join_streamed = !self
                            .streamed_batch
                            .join_filter_matched_idxs
                            .contains(&(self.streamed_batch.idx as u64))
                            && !self.streamed_joined;
                        join_buffered = join_streamed;
                    } else {
                        join_streamed = !self.streamed_joined;
                    }
                }
                if matches!(
                    self.join_type,
                    JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
                ) {
                    join_streamed = true;
                    join_buffered = true;
                };

                if matches!(self.join_type, JoinType::LeftAnti | JoinType::RightAnti)
                    && self.filter.is_some()
                {
                    join_streamed = !self.streamed_joined;
                    join_buffered = join_streamed;
                }
            }
            Ordering::Greater => {
                if matches!(self.join_type, JoinType::Full) {
                    join_buffered = !self.buffered_joined;
                };
            }
        }
        if !join_streamed && !join_buffered {
            self.buffered_data.scanning_finish();
            return Ok(());
        }

        if join_buffered {
            while !self.buffered_data.scanning_finished()
                && self.output_size < self.batch_size
            {
                let scanning_idx = self.buffered_data.scanning_idx();
                if join_streamed {
                    self.streamed_batch.append_output_pair(
                        Some(self.buffered_data.scanning_batch_idx),
                        Some(scanning_idx),
                    );
                } else {
                    self.buffered_data
                        .scanning_batch_mut()
                        .null_joined
                        .push(scanning_idx);
                }
                self.output_size += 1;
                self.buffered_data.scanning_advance();

                if self.buffered_data.scanning_finished() {
                    self.streamed_joined = join_streamed;
                    self.buffered_joined = true;
                }
            }
        } else {
            let scanning_batch_idx = if self.buffered_data.scanning_finished() {
                None
            } else {
                Some(self.buffered_data.scanning_batch_idx)
            };
            let scanning_idx = mark_row_as_match.then_some(0);

            self.streamed_batch
                .append_output_pair(scanning_batch_idx, scanning_idx);
            self.output_size += 1;
            self.buffered_data.scanning_finish();
            self.streamed_joined = true;
        }
        Ok(())
    }

    fn freeze_all(&mut self) -> Result<()> {
        self.freeze_buffered(self.buffered_data.batches.len())?;
        self.freeze_streamed()?;
        Ok(())
    }

    fn freeze_dequeuing_buffered(&mut self) -> Result<()> {
        self.freeze_streamed()?;
        self.freeze_buffered(1)?;
        Ok(())
    }

    fn freeze_buffered(&mut self, batch_count: usize) -> Result<()> {
        if !matches!(self.join_type, JoinType::Full) {
            return Ok(());
        }
        for buffered_batch in self.buffered_data.batches.range_mut(..batch_count) {
            let buffered_indices = UInt64Array::from_iter_values(
                buffered_batch.null_joined.iter().map(|&index| index as u64),
            );
            if let Some(record_batch) = produce_buffered_null_batch(
                &self.schema,
                &self.streamed_schema,
                &buffered_indices,
                buffered_batch,
            )? {
                let num_rows = record_batch.num_rows();
                self.staging_output_record_batches
                    .filter_mask
                    .append_nulls(num_rows);
                self.staging_output_record_batches
                    .row_indices
                    .append_nulls(num_rows);
                self.staging_output_record_batches.batch_ids.resize(
                    self.staging_output_record_batches.batch_ids.len() + num_rows,
                    0,
                );

                self.staging_output_record_batches
                    .batches
                    .push(record_batch);
            }
            buffered_batch.null_joined.clear();
        }
        Ok(())
    }

    fn produce_buffered_not_matched(
        &mut self,
        buffered_batch: &mut BufferedBatch,
    ) -> Result<()> {
        if !matches!(self.join_type, JoinType::Full) {
            return Ok(());
        }

        let not_matched_buffered_indices = buffered_batch
            .join_filter_not_matched_map
            .iter()
            .filter_map(|(idx, failed)| if *failed { Some(*idx) } else { None })
            .collect::<Vec<_>>();

        let buffered_indices =
            UInt64Array::from_iter_values(not_matched_buffered_indices.iter().copied());

        if let Some(record_batch) = produce_buffered_null_batch(
            &self.schema,
            &self.streamed_schema,
            &buffered_indices,
            buffered_batch,
        )? {
            let num_rows = record_batch.num_rows();

            self.staging_output_record_batches
                .filter_mask
                .append_nulls(num_rows);
            self.staging_output_record_batches
                .row_indices
                .append_nulls(num_rows);
            self.staging_output_record_batches.batch_ids.resize(
                self.staging_output_record_batches.batch_ids.len() + num_rows,
                0,
            );
            self.staging_output_record_batches
                .batches
                .push(record_batch);
        }
        buffered_batch.join_filter_not_matched_map.clear();

        Ok(())
    }

    fn freeze_streamed(&mut self) -> Result<()> {
        for chunk in self.streamed_batch.output_indices.iter_mut() {
            let left_indices = chunk.streamed_indices.finish();

            if left_indices.is_empty() {
                continue;
            }

            let mut left_columns = self
                .streamed_batch
                .batch
                .columns()
                .iter()
                .map(|column| take(column, &left_indices, None))
                .collect::<Result<Vec<_>, ArrowError>>()?;

            let right_indices: UInt64Array = chunk.buffered_indices.finish();
            let mut right_columns =
                if matches!(self.join_type, JoinType::LeftMark | JoinType::RightMark) {
                    vec![Arc::new(is_not_null(&right_indices)?) as ArrayRef]
                } else if matches!(
                    self.join_type,
                    JoinType::LeftSemi
                        | JoinType::LeftAnti
                        | JoinType::RightAnti
                        | JoinType::RightSemi
                ) {
                    vec![]
                } else if let Some(buffered_idx) = chunk.buffered_batch_idx {
                    fetch_right_columns_by_idxs(
                        &mut self.buffered_data,
                        buffered_idx,
                        &right_indices,
                    )?
                } else {
                    create_unmatched_columns(
                        self.join_type,
                        &self.buffered_schema,
                        right_indices.len(),
                    )
                };

            let filter_columns = if chunk.buffered_batch_idx.is_some() {
                if !matches!(self.join_type, JoinType::Right) {
                    if matches!(
                        self.join_type,
                        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark
                    ) {
                        let right_cols = fetch_right_columns_by_idxs(
                            &mut self.buffered_data,
                            chunk.buffered_batch_idx.unwrap(),
                            &right_indices,
                        )?;

                        get_filter_column(&self.filter, &left_columns, &right_cols)
                    } else if matches!(
                        self.join_type,
                        JoinType::RightAnti | JoinType::RightSemi | JoinType::RightMark
                    ) {
                        let right_cols = fetch_right_columns_by_idxs(
                            &mut self.buffered_data,
                            chunk.buffered_batch_idx.unwrap(),
                            &right_indices,
                        )?;

                        get_filter_column(&self.filter, &right_cols, &left_columns)
                    } else {
                        get_filter_column(&self.filter, &left_columns, &right_columns)
                    }
                } else {
                    get_filter_column(&self.filter, &right_columns, &left_columns)
                }
            } else {
                vec![]
            };

            let columns = if !matches!(self.join_type, JoinType::Right) {
                left_columns.extend(right_columns);
                left_columns
            } else {
                right_columns.extend(left_columns);
                right_columns
            };

            let output_batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
            if !filter_columns.is_empty() {
                if let Some(f) = &self.filter {
                    let filter_batch =
                        RecordBatch::try_new(Arc::clone(f.schema()), filter_columns)?;

                    let filter_result = f
                        .expression()
                        .evaluate(&filter_batch)?
                        .into_array(filter_batch.num_rows())?;

                    let pre_mask =
                        as_boolean_array(&filter_result)?;

                    let mask = if pre_mask.null_count() > 0 {
                        compute::prep_null_mask_filter(
                            as_boolean_array(&filter_result)?,
                        )
                    } else {
                        pre_mask.clone()
                    };

                    if matches!(
                        self.join_type,
                        JoinType::Left
                            | JoinType::LeftSemi
                            | JoinType::Right
                            | JoinType::RightSemi
                            | JoinType::LeftAnti
                            | JoinType::RightAnti
                            | JoinType::LeftMark
                            | JoinType::RightMark
                            | JoinType::Full
                    ) {
                        self.staging_output_record_batches
                            .batches
                            .push(output_batch);
                    } else {
                        let filtered_batch = filter_record_batch(&output_batch, &mask)?;
                        self.staging_output_record_batches
                            .batches
                            .push(filtered_batch);
                    }

                    if !matches!(self.join_type, JoinType::Full) {
                        self.staging_output_record_batches.filter_mask.extend(&mask);
                    } else {
                        self.staging_output_record_batches
                            .filter_mask
                            .extend(pre_mask);
                    }
                    self.staging_output_record_batches
                        .row_indices
                        .extend(&left_indices);
                    self.staging_output_record_batches.batch_ids.resize(
                        self.staging_output_record_batches.batch_ids.len()
                            + left_indices.len(),
                        self.streamed_batch_counter.load(Relaxed),
                    );

                    if matches!(self.join_type, JoinType::Full) {
                        let buffered_batch = &mut self.buffered_data.batches
                            [chunk.buffered_batch_idx.unwrap()];

                        for i in 0..pre_mask.len() {
                            if right_indices.is_null(i) {
                                continue;
                            }

                            let buffered_index = right_indices.value(i);

                            buffered_batch.join_filter_not_matched_map.insert(
                                buffered_index,
                                *buffered_batch
                                    .join_filter_not_matched_map
                                    .get(&buffered_index)
                                    .unwrap_or(&true)
                                    && !pre_mask.value(i),
                            );
                        }
                    }
                } else {
                    self.staging_output_record_batches
                        .batches
                        .push(output_batch);
                }
            } else {
                self.staging_output_record_batches
                    .batches
                    .push(output_batch);
            }
        }

        self.streamed_batch.output_indices.clear();

        Ok(())
    }

    fn output_record_batch_and_reset(&mut self) -> Result<RecordBatch> {
        let record_batch =
            concat_batches(&self.schema, &self.staging_output_record_batches.batches)?;
        self.join_metrics.output_batches().add(1);
        self.join_metrics
            .baseline_metrics()
            .record_output(record_batch.num_rows());
        if record_batch.num_rows() == 0 || record_batch.num_rows() > self.output_size {
            self.output_size = 0;
        } else {
            self.output_size -= record_batch.num_rows();
        }

        if !(self.filter.is_some()
            && matches!(
                self.join_type,
                JoinType::Left
                    | JoinType::LeftSemi
                    | JoinType::Right
                    | JoinType::RightSemi
                    | JoinType::LeftAnti
                    | JoinType::RightAnti
                    | JoinType::LeftMark
                    | JoinType::RightMark
                    | JoinType::Full
            ))
        {
            self.staging_output_record_batches.batches.clear();
        }

        Ok(record_batch)
    }

    fn filter_joined_batch(&mut self) -> Result<RecordBatch> {
        let record_batch =
            concat_batches(&self.schema, &self.staging_output_record_batches.batches)?;
        let mut out_indices = self.staging_output_record_batches.row_indices.finish();
        let mut out_mask = self.staging_output_record_batches.filter_mask.finish();
        let mut batch_ids = &self.staging_output_record_batches.batch_ids;
        let default_batch_ids = vec![0; record_batch.num_rows()];

        if out_indices.null_count() == out_indices.len()
            && out_indices.len() != record_batch.num_rows()
        {
            out_mask = BooleanArray::from(vec![None; record_batch.num_rows()]);
            out_indices = UInt64Array::from(vec![None; record_batch.num_rows()]);
            batch_ids = &default_batch_ids;
        }

        if out_mask.is_empty() {
            self.staging_output_record_batches.batches.clear();
            return Ok(record_batch);
        }

        let maybe_corrected_mask = get_corrected_filter_mask(
            self.join_type,
            &out_indices,
            batch_ids,
            &out_mask,
            record_batch.num_rows(),
        );

        let corrected_mask = if let Some(ref filtered_join_mask) = maybe_corrected_mask {
            filtered_join_mask
        } else {
            &out_mask
        };

        self.filter_record_batch_by_join_type(record_batch, corrected_mask)
    }

    fn filter_record_batch_by_join_type(
        &mut self,
        record_batch: RecordBatch,
        corrected_mask: &BooleanArray,
    ) -> Result<RecordBatch> {
        let mut filtered_record_batch =
            filter_record_batch(&record_batch, corrected_mask)?;
        let left_columns_length = self.streamed_schema.fields.len();
        let right_columns_length = self.buffered_schema.fields.len();

        if matches!(
            self.join_type,
            JoinType::Left | JoinType::LeftMark | JoinType::Right | JoinType::RightMark
        ) {
            let null_mask = compute::not(corrected_mask)?;
            let null_joined_batch = filter_record_batch(&record_batch, &null_mask)?;

            let mut right_columns = create_unmatched_columns(
                self.join_type,
                &self.buffered_schema,
                null_joined_batch.num_rows(),
            );

            let columns = if !matches!(self.join_type, JoinType::Right) {
                let mut left_columns = null_joined_batch
                    .columns()
                    .iter()
                    .take(right_columns_length)
                    .cloned()
                    .collect::<Vec<_>>();

                left_columns.extend(right_columns);
                left_columns
            } else {
                let left_columns = null_joined_batch
                    .columns()
                    .iter()
                    .skip(left_columns_length)
                    .cloned()
                    .collect::<Vec<_>>();

                right_columns.extend(left_columns);
                right_columns
            };

            let null_joined_streamed_batch =
                RecordBatch::try_new(Arc::clone(&self.schema), columns)?;

            filtered_record_batch = concat_batches(
                &self.schema,
                &[filtered_record_batch, null_joined_streamed_batch],
            )?;
        } else if matches!(self.join_type, JoinType::LeftSemi | JoinType::LeftAnti) {
            let output_column_indices = (0..left_columns_length).collect::<Vec<_>>();
            filtered_record_batch =
                filtered_record_batch.project(&output_column_indices)?;
        } else if matches!(self.join_type, JoinType::RightAnti | JoinType::RightSemi) {
            let output_column_indices = (0..right_columns_length).collect::<Vec<_>>();
            filtered_record_batch =
                filtered_record_batch.project(&output_column_indices)?;
        } else if matches!(self.join_type, JoinType::Full)
            && corrected_mask.false_count() > 0
        {
            let joined_filter_not_matched_mask = compute::not(corrected_mask)?;
            let joined_filter_not_matched_batch =
                filter_record_batch(&record_batch, &joined_filter_not_matched_mask)?;

            let right_null_columns = self
                .buffered_schema
                .fields()
                .iter()
                .map(|f| {
                    new_null_array(
                        f.data_type(),
                        joined_filter_not_matched_batch.num_rows(),
                    )
                })
                .collect::<Vec<_>>();

            let mut result_joined = joined_filter_not_matched_batch
                .columns()
                .iter()
                .take(left_columns_length)
                .cloned()
                .collect::<Vec<_>>();

            result_joined.extend(right_null_columns);

            let left_null_joined_batch =
                RecordBatch::try_new(Arc::clone(&self.schema), result_joined)?;

            let mut result_joined = self
                .streamed_schema
                .fields()
                .iter()
                .map(|f| {
                    new_null_array(
                        f.data_type(),
                        joined_filter_not_matched_batch.num_rows(),
                    )
                })
                .collect::<Vec<_>>();

            let right_data = joined_filter_not_matched_batch
                .columns()
                .iter()
                .skip(left_columns_length)
                .cloned()
                .collect::<Vec<_>>();

            result_joined.extend(right_data);

            filtered_record_batch = concat_batches(
                &self.schema,
                &[filtered_record_batch, left_null_joined_batch],
            )?;
        }

        self.staging_output_record_batches.clear();

        Ok(filtered_record_batch)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn create_unmatched_columns(
    join_type: JoinType,
    schema: &SchemaRef,
    size: usize,
) -> Vec<ArrayRef> {
    if matches!(join_type, JoinType::LeftMark | JoinType::RightMark) {
        vec![Arc::new(BooleanArray::from(vec![false; size])) as ArrayRef]
    } else {
        schema
            .fields()
            .iter()
            .map(|f| new_null_array(f.data_type(), size))
            .collect::<Vec<_>>()
    }
}

/// Gets the arrays which join filters are applied on.
fn get_filter_column(
    join_filter: &Option<JoinFilter>,
    streamed_columns: &[ArrayRef],
    buffered_columns: &[ArrayRef],
) -> Vec<ArrayRef> {
    let mut filter_columns = vec![];

    if let Some(f) = join_filter {
        let left_columns = f
            .column_indices()
            .iter()
            .filter(|col_index| col_index.side == JoinSide::Left)
            .map(|i| Arc::clone(&streamed_columns[i.index]))
            .collect::<Vec<_>>();

        let right_columns = f
            .column_indices()
            .iter()
            .filter(|col_index| col_index.side == JoinSide::Right)
            .map(|i| Arc::clone(&buffered_columns[i.index]))
            .collect::<Vec<_>>();

        filter_columns.extend(left_columns);
        filter_columns.extend(right_columns);
    }

    filter_columns
}

fn produce_buffered_null_batch(
    schema: &SchemaRef,
    streamed_schema: &SchemaRef,
    buffered_indices: &PrimitiveArray<UInt64Type>,
    buffered_batch: &mut BufferedBatch,
) -> Result<Option<RecordBatch>> {
    if buffered_indices.is_empty() {
        return Ok(None);
    }

    let right_columns =
        fetch_right_columns_from_batch_by_idxs(buffered_batch, buffered_indices)?;

    let mut left_columns = streamed_schema
        .fields()
        .iter()
        .map(|f| new_null_array(f.data_type(), buffered_indices.len()))
        .collect::<Vec<_>>();

    left_columns.extend(right_columns);

    Ok(Some(RecordBatch::try_new(
        Arc::clone(schema),
        left_columns,
    )?))
}

/// Get `buffered_indices` rows for `buffered_data[buffered_batch_idx]`
#[inline(always)]
fn fetch_right_columns_by_idxs(
    buffered_data: &mut BufferedData,
    buffered_batch_idx: usize,
    buffered_indices: &UInt64Array,
) -> Result<Vec<ArrayRef>> {
    fetch_right_columns_from_batch_by_idxs(
        &mut buffered_data.batches[buffered_batch_idx],
        buffered_indices,
    )
}

#[inline(always)]
fn fetch_right_columns_from_batch_by_idxs(
    buffered_batch: &mut BufferedBatch,
    buffered_indices: &UInt64Array,
) -> Result<Vec<ArrayRef>> {
    match &buffered_batch.batch {
        BufferedBatchState::InMemory(batch) => Ok(batch
            .columns()
            .iter()
            .map(|column| take(column, &buffered_indices, None))
            .collect::<Result<Vec<_>, ArrowError>>()
            .map_err(Into::<DataFusionError>::into)?),
        BufferedBatchState::Spilled(spill_file) => {
            let batch = if let Some(cached) = &buffered_batch.spill_cache {
                cached.clone()
            } else {
                let file = BufReader::new(File::open(spill_file.path())?);
                let reader = StreamReader::try_new(file, None)?;

                let mut batches = Vec::new();
                for b in reader {
                    batches.push(b?);
                }
                let batch = concat_batches(&batches[0].schema(), &batches)?;
                buffered_batch.spill_cache = Some(batch.clone());
                batch
            };

            Ok(batch
                .columns()
                .iter()
                .map(|column| take(column, &buffered_indices, None))
                .collect::<Result<Vec<_>, ArrowError>>()
                .map_err(Into::<DataFusionError>::into)?)
        }
    }
}

/// Buffered data contains all buffered batches with one unique join key
#[derive(Debug, Default)]
struct BufferedData {
    pub batches: VecDeque<BufferedBatch>,
    pub scanning_batch_idx: usize,
    pub scanning_offset: usize,
}

impl BufferedData {
    fn head_batch(&self) -> &BufferedBatch {
        self.batches.front().unwrap()
    }

    fn tail_batch(&self) -> &BufferedBatch {
        self.batches.back().unwrap()
    }

    fn tail_batch_mut(&mut self) -> &mut BufferedBatch {
        self.batches.back_mut().unwrap()
    }

    fn has_buffered_rows(&self) -> bool {
        self.batches.iter().any(|batch| !batch.range.is_empty())
    }

    fn scanning_reset(&mut self) {
        self.scanning_batch_idx = 0;
        self.scanning_offset = 0;
    }

    fn scanning_advance(&mut self) {
        self.scanning_offset += 1;
        while !self.scanning_finished() && self.scanning_batch_finished() {
            self.scanning_batch_idx += 1;
            self.scanning_offset = 0;
        }
    }

    fn scanning_batch(&self) -> &BufferedBatch {
        &self.batches[self.scanning_batch_idx]
    }

    fn scanning_batch_mut(&mut self) -> &mut BufferedBatch {
        &mut self.batches[self.scanning_batch_idx]
    }

    fn scanning_idx(&self) -> usize {
        self.scanning_batch().range.start + self.scanning_offset
    }

    fn scanning_batch_finished(&self) -> bool {
        self.scanning_offset == self.scanning_batch().range.len()
    }

    fn scanning_finished(&self) -> bool {
        self.scanning_batch_idx == self.batches.len()
    }

    fn scanning_finish(&mut self) {
        self.scanning_batch_idx = self.batches.len();
        self.scanning_offset = 0;
    }
}

/// Get join array refs of given batch and join columns
fn join_arrays(batch: &RecordBatch, on_column: &[PhysicalExprRef]) -> Vec<ArrayRef> {
    on_column
        .iter()
        .map(|c| {
            let num_rows = batch.num_rows();
            let c = c.evaluate(batch).unwrap();
            c.into_array(num_rows).unwrap()
        })
        .collect()
}

/// Vectorized key-group boundary detection.
///
/// Returns the number of consecutive rows starting from `start` in `target_arrays`
/// that have the same key as `ref_arrays[ref_idx]`. Uses Arrow's SIMD-accelerated
/// `eq` kernel instead of row-by-row comparison with runtime type dispatch.
///
/// Null semantics: NULL == NULL for grouping purposes (matching Spark's behavior
/// for buffered-side key grouping). Non-null vs null is always unequal.
fn find_key_group_boundary(
    ref_arrays: &[ArrayRef],
    ref_idx: usize,
    target_arrays: &[ArrayRef],
    start: usize,
    num_rows: usize,
) -> Result<usize> {
    let len = num_rows - start;
    if len == 0 {
        return Ok(0);
    }

    let mut combined_mask: Option<BooleanArray> = None;

    for (ref_arr, target_arr) in ref_arrays.iter().zip(target_arrays) {
        let target_slice = target_arr.slice(start, len);

        let col_mask = if ref_arr.is_null(ref_idx) {
            // NULL == NULL for grouping purposes
            is_null(&target_slice)?
        } else {
            let ref_scalar = Scalar::new(ref_arr.slice(ref_idx, 1));
            let eq_result = eq(&target_slice, &ref_scalar)?;
            // eq returns NULL where target is null; convert nulls to false
            if eq_result.null_count() > 0 {
                let values = eq_result.values() & eq_result.nulls().unwrap().inner();
                BooleanArray::new(values, None)
            } else {
                eq_result
            }
        };

        combined_mask = Some(match combined_mask {
            None => col_mask,
            Some(prev) => and(&prev, &col_mask)?,
        });
    }

    let mask = combined_mask.unwrap();
    // Find position of first false (end of matching group)
    let count = mask.values().iter().take_while(|&v| v).count();
    Ok(count)
}
