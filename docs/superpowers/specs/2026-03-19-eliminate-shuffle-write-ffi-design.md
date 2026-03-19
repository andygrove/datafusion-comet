# Eliminate FFI Round-Trip from Native Shuffle Write Path

## Problem

When a native child plan feeds into native shuffle write, data crosses the JVM/native boundary twice unnecessarily:

1. **FFI #1 (native to JVM)**: `CometNativeExec.doExecuteColumnar()` executes the child plan natively and returns `ColumnarBatch` to JVM via Arrow FFI.
2. **FFI #2 (JVM to native)**: `CometNativeShuffleWriter` creates a standalone `ShuffleWriter -> Scan` native plan. The `ScanExec` reads those same JVM batches back via FFI.

The child plan's output is produced natively, materialized in JVM, then immediately consumed natively again. The JVM never inspects or transforms these batches -- it's pure overhead.

## Solution

Fold the child's native plan into the shuffle writer's native plan, creating a single combined plan: `ShuffleWriter -> <child operators> -> Scan`. The shuffle writer's input becomes the child's raw input (e.g., file scan output), not the child's processed output. This eliminates one FFI round-trip.

### Scope

- **Write side only.** Read-side optimization is a separate PR.
- **Single-input child plans only.** Plans with multiple input sources (e.g., joins with two scan children) or broadcast inputs fall back to current double-FFI behavior.
- **CometNativeShuffle only.** CometColumnarShuffle is unaffected.

## Data Flow

### Before (double FFI)

```
CometNativeExec.doExecuteColumnar()
  -> CometExecRDD -> CometExecIterator
    -> Native: Scan(file) -> Filter -> Project
    -> Arrow FFI -> JVM ColumnarBatch              [FFI #1]

CometShuffleExchangeExec.inputRDD = child.executeColumnar()
  -> CometShuffleDependency(rdd = outputBatches.map((0, _)))

CometNativeShuffleWriter.write(outputBatches)
  -> CometExecIterator(plan = ShuffleWriter -> Scan)
    -> Native: Scan reads outputBatches via FFI    [FFI #2]
    -> ShuffleWriterExec partitions & writes to disk
```

### After (single FFI)

```
CometShuffleExchangeExec detects native child with single input
  -> Extracts child's serialized plan + raw input RDD
  -> CometShuffleDependency(rdd = rawInputBatches.map((0, _)),
                            childNativePlan = childPlanBytes,
                            childNativeMetrics = childMetricsTree)

CometNativeShuffleWriter.write(rawInputBatches)
  -> CometExecIterator(plan = ShuffleWriter -> Filter -> Project -> Scan,
                       metrics = combined metrics tree)
    -> Native: Scan reads rawInputBatches via FFI  [only FFI]
    -> Filter -> Project (all native, no JVM)
    -> ShuffleWriterExec partitions & writes to disk
```

## Eligibility Criteria

The optimization applies when ALL conditions hold:

1. `shuffleType == CometNativeShuffle`
2. Child is a `CometNativeExec` with a defined `serializedPlanOpt`
3. `foreachUntilCometInput(child)` finds exactly ONE input source
4. That input source is not a broadcast (not `CometBroadcastExchangeExec` or `BroadcastQueryStageExec`)
5. Output partitioning is NOT `RangePartitioning` (see note below)
6. Child plan has no per-partition plan data (no `CometIcebergNativeScanExec` with non-empty `perPartitionData`)
7. Child plan has no encrypted Parquet scans (`CometNativeScanExec` with encryption enabled)
8. Child plan has no scalar subqueries

When any condition fails, fall back to `child.executeColumnar()` (current behavior).

### Why exclude RangePartitioning?

`prepareShuffleDependency` samples batches from the input RDD using `outputAttributes` to compute partition bounds. With the optimization, the input RDD contains *raw* batches (pre-Filter/Project), whose schema differs from `outputAttributes`. The sampling projection would fail or produce incorrect bounds. A future enhancement could separate the sampling RDD from the shuffle RDD, but for now we exclude this case. `HashPartitioning`, `SinglePartition`, and `RoundRobinPartitioning` do not sample the RDD and are unaffected.

### Why exclude Iceberg/encrypted/subquery plans?

`CometNativeExec.doExecuteColumnar()` performs per-partition plan data injection for Iceberg scans, collects encryption configs for encrypted Parquet, and registers scalar subqueries. The combined path bypasses `doExecuteColumnar()`, so these features would not work. Rather than replicating that logic in the shuffle writer, we exclude these cases for now. These are relatively uncommon compared to the standard Scan -> operators -> Shuffle pattern.

## Changes by File

### CometShuffleExchangeExec.scala

Modify `inputRDD` (or introduce a helper) to detect eligible native children:

```scala
@transient lazy val inputRDD: RDD[_] = if (shuffleType == CometNativeShuffle) {
  child match {
    case nativeChild: CometNativeExec if canCombineWithShuffleWriter(nativeChild) =>
      // Use child's raw input RDD instead of child's output
      collectSingleInput(nativeChild)
    case _ =>
      child.executeColumnar()
  }
} else if (shuffleType == CometColumnarShuffle) {
  child.execute()
} else {
  throw new UnsupportedOperationException(...)
}
```

Add `canCombineWithShuffleWriter(child: CometNativeExec): Boolean`:
- Check `serializedPlanOpt.isDefined`
- Check `outputPartitioning` is not `RangePartitioning`
- Use `foreachUntilCometInput` to count input sources; return false if != 1 or if broadcast
- Check no `CometIcebergNativeScanExec` with per-partition data in child tree
- Check no encrypted Parquet scans in child tree
- Check no scalar subqueries in child plan expressions
- Check config flag `spark.comet.exec.shuffle.combinePlans.enabled` is true

Add `collectSingleInput(child: CometNativeExec): RDD[ColumnarBatch]`:
- Walk child plan with `foreachUntilCometInput` to find the single input source
- Call `executeColumnar()` on that input source to get the raw input RDD

Modify `prepareShuffleDependency` (or the call site) to pass `childNativePlan` and `childNativeMetrics` when combining:
- `childNativePlan = Some(nativeChild.serializedPlanOpt.plan.get)`
- `childNativeMetrics = Some(CometMetricNode.fromCometPlan(nativeChild))`

### CometShuffleDependency.scala

Add two optional fields:

```scala
class CometShuffleDependency[K: ClassTag, V: ClassTag, C: ClassTag](
    ...existing fields...
    val childNativePlan: Option[Array[Byte]] = None,
    val childNativeMetrics: Option[CometMetricNode] = None)
```

### CometShuffleManager.scala

In `getWriter`, pass the new fields from the dependency to `CometNativeShuffleWriter`:

```scala
case cometShuffleHandle: CometNativeShuffleHandle[K, V] =>
  val dep = cometShuffleHandle.dependency.asInstanceOf[CometShuffleDependency[_, _, _]]
  new CometNativeShuffleWriter(
    ...existing args...
    dep.childNativePlan,
    dep.childNativeMetrics)
```

### CometNativeShuffleWriter.scala

Add constructor parameters:

```scala
class CometNativeShuffleWriter[K, V](
    ...existing params...
    childNativePlan: Option[Array[Byte]] = None,
    childNativeMetrics: Option[CometMetricNode] = None)
```

Modify `getNativePlan()`: when `childNativePlan` is provided, deserialize it and use it as the ShuffleWriter's child instead of a bare Scan:

```scala
private def getNativePlan(dataFile: String, indexFile: String): Operator = {
  ...build shuffleWriterBuilder as before...

  if (childNativePlan.isDefined) {
    // Combined plan: ShuffleWriter -> child plan (which already has Scan leaves)
    val childOp = OperatorOuterClass.Operator.parseFrom(childNativePlan.get)
    shuffleWriterOpBuilder
      .setShuffleWriter(shuffleWriterBuilder)
      .addChildren(childOp)
      .build()
  } else {
    // Current behavior: ShuffleWriter -> Scan
    shuffleWriterOpBuilder
      .setShuffleWriter(shuffleWriterBuilder)
      .addChildren(opBuilder.setScan(scanBuilder).build())
      .build()
  }
}
```

Modify `write()` to build combined metrics:

```scala
val nativeMetrics = if (childNativeMetrics.isDefined) {
  // Combined metrics tree: shuffle writer metrics with child metrics as children
  CometMetricNode(nativeSQLMetrics, Seq(childNativeMetrics.get))
} else {
  CometMetricNode(nativeSQLMetrics)
}
```

## What Does NOT Change

- **Rust/native code**: `PhysicalPlanner` already handles `ShuffleWriter` with arbitrary child plans. It recursively calls `create_plan` on children. No native changes needed.
- **Shuffle read path**: Unchanged (separate PR).
- **CometColumnarShuffle**: Unchanged.
- **Protobuf definitions**: Unchanged. The existing `ShuffleWriter` message already supports arbitrary child operators.
- **Multi-input plans**: Fall back to current behavior. No regression.

## Metrics

The combined `CometMetricNode` tree mirrors the combined native plan structure. Native code walks the metrics tree in parallel with the operator tree, updating each operator's `SQLMetric` via JNI. Since `SQLMetric` extends `AccumulatorV2`, values serialize to executors and report back to the driver correctly.

The Spark UI will show the same per-operator metrics (output rows, elapsed time) as before for child plan operators.

## Testing

- Existing shuffle tests should continue to pass (they exercise the eligible path).
- Verify via Spark UI or metrics that child plan operator metrics are populated.
- Test with multi-input child plans (joins) to verify fallback works.
- Test with non-native children to verify fallback works.
- Consider adding a test that explicitly checks the plan structure (combined vs. standalone).

## Configuration

Add `spark.comet.exec.shuffle.combinePlans.enabled` (default: `true`) in `CometConf.scala`. When false, always falls back to current behavior. This provides a safe rollback mechanism if issues arise.

## Future Work

- Extend to multi-input child plans (joins before shuffle) by threading multiple input RDDs.
- Support `RangePartitioning` by separating the sampling RDD from the shuffle dependency RDD.
- Support Iceberg scans by threading per-partition plan data injection into the shuffle writer.
- Support encrypted Parquet scans by threading encryption configs through the dependency.
- Support scalar subqueries in child plans by registering them in the shuffle writer.
- Combine with read-side optimization (separate PR) for end-to-end FFI elimination.
- Handle broadcast inputs in the combined path.
