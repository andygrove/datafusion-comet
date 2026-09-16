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

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::sync::atomic::{AtomicBool, Ordering};

use datafusion::{
    common::DataFusionError,
    execution::memory_pool::{MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation},
};
use log::warn;

use crate::alloc_accounting;

/// Wraps an off-heap memory pool and reports when the bytes the native allocator has actually
/// handed out, plus a pending request, would exceed `spark.memory.offHeap.size`.
///
/// A pool on its own only counts what operators voluntarily reserve. Native memory that bypasses
/// it, such as scratch buffers inside kernels or intermediate Arrow arrays, is invisible until the
/// executor exceeds its container limit and is killed. This wrapper consults the
/// `alloc-accounting` balance on every reservation so that the gap between declared and real usage
/// is visible at runtime rather than only in a post-mortem.
///
/// **It only observes.** The reservation is always passed to the inner pool. An earlier revision
/// could also refuse the reservation, and that was removed because it was measured not to work:
/// on TPC-H SF100 at a 2 GB budget it refused 30 to 44 reservations per run and real native usage
/// still passed the budget in every run, exactly as it did when only observing. The reason is
/// structural. The gate refuses *reservations*, but the overshoot lives in allocations that never
/// reserve, so making a reserving operator spill releases *reserved* bytes and does nothing about
/// the untracked allocations actually carrying usage past the limit. Refusing therefore penalised
/// the operators that play by the rules without relieving the pressure.
///
/// The budget is `spark.memory.offHeap.size`, deliberately not the pool's own limit: that limit is
/// the off-heap size times `spark.comet.exec.memoryPool.fraction`, and the fraction is how
/// operators hold back reservable memory to force spilling. Deriving this budget from it too would
/// make a small fraction report a crossing on every reservation.
///
/// Note the two sides do not measure the same population: `spark.memory.offHeap.size` is shared
/// with Spark's own Tungsten off-heap allocations and with Comet's JVM-side shuffle pages, while
/// the balance counts only Comet's Rust allocations. The comparison is a signal that Comet is
/// heading past its allotment, not a bound on total off-heap usage.
///
/// Both the balance and the budget are process-wide, so there is no per-task attribution: once any
/// task pushes real usage past the budget, every task's next non-zero reservation sees it.
///
/// A build without the `alloc-accounting` feature reports a balance of zero, which leaves the
/// wrapper a passthrough.
pub struct NativeUsagePool<P: MemoryPool> {
    inner: P,
    budget: usize,
    /// Set once this pool has logged, so a crossing does not flood the log: `try_grow` is called
    /// constantly and the condition is sticky once real usage is high. One line per pool means
    /// one line per task that reached the budget.
    reported: AtomicBool,
}

impl<P: MemoryPool> NativeUsagePool<P> {
    pub fn new(inner: P, budget: usize) -> Self {
        Self {
            inner,
            budget,
            reported: AtomicBool::new(false),
        }
    }
}

impl<P: MemoryPool> Debug for NativeUsagePool<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("NativeUsagePool")
            .field("budget", &self.budget)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<P: MemoryPool> Display for NativeUsagePool<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "NativeUsagePool(budget={}, inner={})",
            self.budget, self.inner
        )
    }
}

impl<P: MemoryPool> MemoryPool for NativeUsagePool<P> {
    fn name(&self) -> &str {
        "NativeUsagePool"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer)
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer)
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.try_grow(reservation, additional).unwrap()
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink)
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> Result<(), DataFusionError> {
        if additional == 0 {
            return Ok(());
        }
        // A single relaxed atomic load, so this sits in front of the inner pool's JNI round trip
        // to Spark without adding a measurable cost.
        let in_use = alloc_accounting::current_balance();
        if in_use.saturating_add(additional) > self.budget && !self.reported.swap(true, Ordering::Relaxed)
        {
            // Both quantities are named because either can be what crosses the budget: a large
            // single request against modest usage, or a small request against usage already near
            // the ceiling.
            warn!(
                "Reserving {additional} bytes for {} would take Comet's real native memory usage \
                 ({in_use} bytes) past spark.memory.offHeap.size ({} bytes). This memory is not \
                 covered by the pool's reservations, so the executor may exceed its container \
                 limit even though Spark's accounting looks healthy. Consider raising \
                 spark.memory.offHeap.size or lowering spark.comet.exec.memoryPool.fraction.",
                reservation.consumer().name(),
                self.budget
            );
        }
        self.inner.try_grow(reservation, additional)
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        // Always the inner pool's limit, never the budget. The budget is a process-wide signal
        // compared against process-wide allocator usage, not this pool's reservable limit.
        // DataFusion also acts on this value: `AggregateExec::should_use_partial_reduce_hash_stream`
        // bails out whenever a pool reports `Finite`, so returning the budget here would silently
        // change the aggregation strategy as a side effect of observing memory.
        self.inner.memory_limit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::memory_pool::UnboundedMemoryPool;
    use std::sync::Arc;

    fn pool(budget: usize) -> Arc<dyn MemoryPool> {
        Arc::new(NativeUsagePool::new(UnboundedMemoryPool::default(), budget))
    }

    #[test]
    fn a_zero_byte_grow_never_fails() {
        let p = pool(0);
        let reservation = MemoryConsumer::new("zero").register(&p);
        reservation.try_grow(0).unwrap();
    }

    /// Never `Finite(budget)`. DataFusion changes its aggregation strategy when a pool reports a
    /// finite limit, so reporting the budget would make observing memory alter execution.
    #[test]
    fn always_reports_the_inner_limit() {
        assert!(matches!(pool(4096).memory_limit(), MemoryLimit::Infinite));
    }

    #[test]
    fn grows_and_shrinks_reach_the_inner_pool() {
        let p = pool(usize::MAX);
        let reservation = MemoryConsumer::new("delegate").register(&p);
        reservation.try_grow(1024).unwrap();
        assert_eq!(p.reserved(), 1024);
        reservation.shrink(1024);
        assert_eq!(p.reserved(), 0);
    }

    /// The whole point of the wrapper being observe-only: a request that takes real usage past the
    /// budget still succeeds and still reaches the inner pool.
    ///
    /// The block is touched so it is really allocated, and the margins are far wider than anything
    /// the rest of the crate allocates in the microseconds between the checks. The serial lock
    /// keeps the accounting tests that move the balance by tens of megabytes out of that window.
    #[test]
    #[cfg(feature = "alloc-accounting")]
    fn a_crossing_is_observed_but_never_refused() {
        use std::hint::black_box;

        const HEADROOM: usize = 64 * 1024 * 1024;
        const BLOCK: usize = 256 * 1024 * 1024;

        let _guard = alloc_accounting::test_support::serial();
        let budget = alloc_accounting::current_balance() + HEADROOM;
        let p = pool(budget);
        let reservation = MemoryConsumer::new("observed").register(&p);

        let held: Vec<u8> = black_box(vec![1u8; BLOCK]);
        reservation.try_grow(1).unwrap();
        black_box(&held);
        assert_eq!(p.reserved(), 1, "observing must not withhold the bytes");
        drop(held);
    }
}
