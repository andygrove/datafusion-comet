# Eliminate Shuffle Write FFI Round-Trip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the unnecessary FFI round-trip when a native child plan feeds into native shuffle write, by folding the child's native plan into the shuffle writer's native plan.

**Architecture:** When the child of a `CometShuffleExchangeExec` is a `CometNativeExec` with a single input source, pass the child's serialized native plan and metrics through `CometShuffleDependency` to `CometNativeShuffleWriter`. The writer builds a combined `ShuffleWriter -> <child plan>` instead of `ShuffleWriter -> Scan`, feeding raw input batches directly. Ineligible plans (multi-input, RangePartitioning, Iceberg, encrypted, subqueries) fall back to current behavior.

**Tech Stack:** Scala (Spark plugin), Protobuf (plan serialization), Rust/DataFusion (native execution — no changes needed)

**Spec:** `docs/superpowers/specs/2026-03-19-eliminate-shuffle-write-ffi-design.md`

---

### Task 1: Add config flag in CometConf

**Files:**
- Modify: `common/src/main/scala/org/apache/comet/CometConf.scala`

- [ ] **Step 1: Add the config entry**

Add after `COMET_EXEC_SHUFFLE_ENABLED` (around line 344):

```scala
val COMET_EXEC_SHUFFLE_COMBINE_PLANS_ENABLED: ConfigEntry[Boolean] =
  conf(s"$COMET_EXEC_CONFIG_PREFIX.shuffle.combinePlans.enabled")
    .category(CATEGORY_SHUFFLE)
    .doc(
      "When enabled, folds a native child plan into the shuffle writer's native plan, " +
        "eliminating an FFI round-trip. Only applies when the child plan has a single " +
        "non-broadcast input source and does not use RangePartitioning, Iceberg scans, " +
        "encrypted Parquet, or scalar subqueries.")
    .booleanConf
    .createWithDefault(true)
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw compile -pl common -q`
Expected: BUILD SUCCESS

- [ ] **Step 3: Commit**

```bash
git add common/src/main/scala/org/apache/comet/CometConf.scala
git commit -m "feat: add config flag for shuffle write plan combining"
```

---

### Task 2: Add fields to CometShuffleDependency

**Files:**
- Modify: `spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleDependency.scala`

- [ ] **Step 1: Add the two optional fields**

Add `childNativePlan` and `childNativeMetrics` parameters to `CometShuffleDependency`, before the closing `)` of the constructor parameter list (after `rangePartitionBounds` at line 52). Also add the import for `CometMetricNode`.

```scala
import org.apache.spark.sql.comet.CometMetricNode
```

Add parameters:
```scala
    val childNativePlan: Option[Array[Byte]] = None,
    val childNativeMetrics: Option[CometMetricNode] = None)
```

- [ ] **Step 2: Build to verify compilation**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw compile -pl spark -q`
Expected: BUILD SUCCESS

- [ ] **Step 3: Commit**

```bash
git add spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleDependency.scala
git commit -m "feat: add child native plan fields to CometShuffleDependency"
```

---

### Task 3: Thread child plan through CometShuffleManager to CometNativeShuffleWriter

**Files:**
- Modify: `spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleManager.scala:230-241`
- Modify: `spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometNativeShuffleWriter.scala:47-57`

- [ ] **Step 1: Add constructor parameters to CometNativeShuffleWriter**

Add two parameters after `rangePartitionBounds` (line 56 of `CometNativeShuffleWriter.scala`):

```scala
    childNativePlan: Option[Array[Byte]] = None,
    childNativeMetrics: Option[CometMetricNode] = None)
```

Add the import at the top of the file:
```scala
import org.apache.spark.sql.comet.CometMetricNode
```

- [ ] **Step 2: Pass fields from CometShuffleManager.getWriter**

In `CometShuffleManager.scala`, modify the `CometNativeShuffleHandle` case (lines 230-241) to pass the new fields:

```scala
      case cometShuffleHandle: CometNativeShuffleHandle[K @unchecked, V @unchecked] =>
        val dep = cometShuffleHandle.dependency.asInstanceOf[CometShuffleDependency[_, _, _]]
        new CometNativeShuffleWriter(
          dep.outputPartitioning.get,
          dep.outputAttributes,
          dep.shuffleWriteMetrics,
          dep.numParts,
          dep.shuffleId,
          mapId,
          context,
          metrics,
          dep.rangePartitionBounds,
          dep.childNativePlan,
          dep.childNativeMetrics)
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw compile -pl spark -q`
Expected: BUILD SUCCESS

- [ ] **Step 4: Commit**

```bash
git add spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleManager.scala \
        spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometNativeShuffleWriter.scala
git commit -m "feat: thread child native plan from dependency to shuffle writer"
```

---

### Task 4: Modify CometNativeShuffleWriter to build combined plan and metrics

**Files:**
- Modify: `spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometNativeShuffleWriter.scala`

- [ ] **Step 1: Modify getNativePlan() to use child plan when available**

Replace the final block of `getNativePlan()` (lines 310-314) where the ShuffleWriter operator is built. Currently:

```scala
      val shuffleWriterOpBuilder = OperatorOuterClass.Operator.newBuilder()
      shuffleWriterOpBuilder
        .setShuffleWriter(shuffleWriterBuilder)
        .addChildren(opBuilder.setScan(scanBuilder).build())
        .build()
```

Replace with:

```scala
      val shuffleWriterOpBuilder = OperatorOuterClass.Operator.newBuilder()
      shuffleWriterOpBuilder.setShuffleWriter(shuffleWriterBuilder)

      if (childNativePlan.isDefined) {
        // Combined plan: ShuffleWriter -> <child native plan>
        // The child plan already has Scan leaves that will read from the input iterators.
        val childOp = OperatorOuterClass.Operator.parseFrom(childNativePlan.get)
        shuffleWriterOpBuilder.addChildren(childOp).build()
      } else {
        // Original behavior: ShuffleWriter -> Scan
        shuffleWriterOpBuilder
          .addChildren(opBuilder.setScan(scanBuilder).build())
          .build()
      }
```

Note: When `childNativePlan` is provided, `scanBuilder` and `scanTypes` are still built but unused. This is fine — the code path that checks `scanTypes.length == outputAttributes.length` (line 173) must still pass to reach the ShuffleWriter builder. The `outputAttributes` still represent the shuffle output schema (used for partitioning expressions), not the scan schema. Since `childNativePlan` is only set when the child is native (meaning all types are supported), this check will pass.

- [ ] **Step 2: Modify write() to build combined metrics tree**

In the `write()` method (around line 94), replace:

```scala
    val nativeMetrics = CometMetricNode(nativeSQLMetrics)
```

With:

```scala
    val nativeMetrics = childNativeMetrics match {
      case Some(childMetrics) =>
        // Combined metrics tree: shuffle writer metrics as root, child metrics as subtree.
        // This mirrors the combined native plan structure so native code can walk both
        // trees in parallel to update per-operator metrics.
        CometMetricNode(nativeSQLMetrics, Seq(childMetrics))
      case None =>
        CometMetricNode(nativeSQLMetrics)
    }
```

- [ ] **Step 3: Build to verify compilation**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw compile -pl spark -q`
Expected: BUILD SUCCESS

- [ ] **Step 4: Commit**

```bash
git add spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometNativeShuffleWriter.scala
git commit -m "feat: build combined native plan and metrics in shuffle writer"
```

---

### Task 5: Add eligibility check and input collection in CometShuffleExchangeExec

This is the core change. Adds `canCombineWithShuffleWriter` and `collectSingleInput`, and modifies `inputRDD` and `prepareShuffleDependency` to use them.

**Files:**
- Modify: `spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleExchangeExec.scala`

- [ ] **Step 1: Add imports**

Add these imports to the file (check existing imports first — some may already be present):

```scala
import org.apache.spark.sql.comet.{CometBroadcastExchangeExec, CometIcebergNativeScanExec,
  CometMetricNode, CometNativeExec, CometNativeScanExec, CometPlan, CometSinkPlaceHolder}
import org.apache.spark.sql.execution.ScalarSubquery
import org.apache.spark.sql.execution.adaptive.BroadcastQueryStageExec
```

Note: `CometNativeExec`, `CometSinkPlaceHolder`, `CometMetricNode`, and `CometPlan` are already imported on line 37. `ShuffleQueryStageExec` is imported on line 39 but `BroadcastQueryStageExec` is NOT — it must be added. Merge into existing import groups rather than adding duplicate imports.

- [ ] **Step 2: Add helper methods to the companion object**

Add these methods inside the `CometShuffleExchangeExec` companion object (after `prepareShuffleDependency`, around line 637):

```scala
  /**
   * Checks if a native child plan is eligible to be folded into the shuffle writer's
   * native plan, eliminating an FFI round-trip. See the design spec for full criteria.
   */
  private[shuffle] def canCombineWithShuffleWriter(
      nativeChild: CometNativeExec,
      outputPartitioning: Partitioning): Boolean = {

    if (!CometConf.COMET_EXEC_SHUFFLE_COMBINE_PLANS_ENABLED.get()) return false
    if (!nativeChild.serializedPlanOpt.isDefined) return false

    // Exclude RangePartitioning: prepareShuffleDependency samples the inputRDD using
    // outputAttributes, which won't match the raw input schema.
    outputPartitioning match {
      case _: RangePartitioning => return false
      case _ =>
    }

    // Count input sources and check for broadcasts
    val inputSources = scala.collection.mutable.ArrayBuffer.empty[SparkPlan]
    nativeChild.foreachUntilCometInput(nativeChild)(inputSources += _)

    if (inputSources.size != 1) return false

    // Reject broadcast inputs
    inputSources.head match {
      case _: CometBroadcastExchangeExec | _: BroadcastQueryStageExec => return false
      case _ =>
    }

    // Reject Iceberg scans with per-partition data
    var hasIceberg = false
    nativeChild.foreachUntilCometInput(nativeChild) {
      case ice: CometIcebergNativeScanExec if ice.perPartitionData.nonEmpty =>
        hasIceberg = true
      case _ =>
    }
    if (hasIceberg) return false

    // Reject encrypted Parquet scans
    var hasEncrypted = false
    nativeChild.foreachUntilCometInput(nativeChild) {
      case scan: CometNativeScanExec =>
        val hadoopConf = scan.relation.sparkSession.sessionState
          .newHadoopConfWithOptions(scan.relation.options)
        if (CometParquetUtils.encryptionEnabled(hadoopConf)) {
          hasEncrypted = true
        }
      case _ =>
    }
    if (hasEncrypted) return false

    // Reject plans with scalar subqueries
    val hasSubqueries = nativeChild.expressions.exists(
      _.find(_.isInstanceOf[ScalarSubquery]).isDefined)
    if (hasSubqueries) return false

    // Also check child plan tree for subqueries (not just the boundary node)
    var childHasSubqueries = false
    def checkSubqueries(plan: SparkPlan): Unit = plan match {
      case _: CometNativeExec =>
        if (plan.expressions.exists(_.find(_.isInstanceOf[ScalarSubquery]).isDefined)) {
          childHasSubqueries = true
        }
        plan.children.foreach(checkSubqueries)
      case _ => // stop at non-native nodes
    }
    nativeChild.children.foreach(checkSubqueries)
    if (childHasSubqueries) return false

    true
  }

  /**
   * Collects the single input RDD from a native child plan. The caller must have already
   * verified eligibility via canCombineWithShuffleWriter.
   */
  private[shuffle] def collectSingleInput(
      nativeChild: CometNativeExec): RDD[ColumnarBatch] = {
    val inputSources = scala.collection.mutable.ArrayBuffer.empty[SparkPlan]
    nativeChild.foreachUntilCometInput(nativeChild)(inputSources += _)
    assert(inputSources.size == 1,
      s"Expected exactly one input source, found ${inputSources.size}")
    inputSources.head.executeColumnar()
  }
```

Also add the necessary import for `CometParquetUtils` if not present:
```scala
import org.apache.comet.parquet.CometParquetUtils
```

- [ ] **Step 3: Modify inputRDD to detect eligible native children**

Replace the `inputRDD` lazy val (lines 92-103):

```scala
  @transient lazy val inputRDD: RDD[_] = if (shuffleType == CometNativeShuffle) {
    child match {
      case nativeChild: CometNativeExec
          if CometShuffleExchangeExec.canCombineWithShuffleWriter(
            nativeChild, outputPartitioning) =>
        CometShuffleExchangeExec.collectSingleInput(nativeChild)
      case _ =>
        child.executeColumnar()
    }
  } else if (shuffleType == CometColumnarShuffle) {
    child.execute()
  } else {
    throw new UnsupportedOperationException(
      s"Unsupported shuffle type: ${shuffleType.getClass.getName}")
  }
```

- [ ] **Step 4: Add a method to extract child plan info and pass through dependency**

Add a helper to check if we're in combined mode and extract the plan/metrics. Add to the `CometShuffleExchangeExec` case class body (e.g., after `inputRDD`):

```scala
  /** Returns (childNativePlan, childNativeMetrics) if in combined mode, else (None, None). */
  @transient private lazy val childPlanInfo: (Option[Array[Byte]], Option[CometMetricNode]) = {
    child match {
      case nativeChild: CometNativeExec
          if shuffleType == CometNativeShuffle &&
            CometShuffleExchangeExec.canCombineWithShuffleWriter(
              nativeChild, outputPartitioning) =>
        (nativeChild.serializedPlanOpt.plan, Some(CometMetricNode.fromCometPlan(nativeChild)))
      case _ =>
        (None, None)
    }
  }
```

- [ ] **Step 5: Pass child plan info through prepareShuffleDependency**

Modify `prepareShuffleDependency` signature (line 563) to accept the new optional parameters:

```scala
  def prepareShuffleDependency(
      rdd: RDD[ColumnarBatch],
      outputAttributes: Seq[Attribute],
      outputPartitioning: Partitioning,
      serializer: Serializer,
      metrics: Map[String, SQLMetric],
      childNativePlan: Option[Array[Byte]] = None,
      childNativeMetrics: Option[CometMetricNode] = None): ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {
```

Add the new fields to the `CometShuffleDependency` constructor (around line 622-636):

```scala
    val dependency = new CometShuffleDependency[Int, ColumnarBatch, ColumnarBatch](
      rdd.map(
        (0, _)
      ),
      serializer = serializer,
      shuffleWriterProcessor = ShuffleExchangeExec.createShuffleWriteProcessor(metrics),
      shuffleType = CometNativeShuffle,
      partitioner = partitioner,
      decodeTime = metrics("decode_time"),
      outputPartitioning = Some(outputPartitioning),
      outputAttributes = outputAttributes,
      shuffleWriteMetrics = metrics,
      numParts = numParts,
      rangePartitionBounds = rangePartitionBounds,
      childNativePlan = childNativePlan,
      childNativeMetrics = childNativeMetrics)
```

- [ ] **Step 6: Update the call site in shuffleDependency**

Modify the `shuffleDependency` lazy val (lines 138-152) to pass `childPlanInfo`:

```scala
  @transient
  lazy val shuffleDependency: ShuffleDependency[Int, _, _] =
    if (shuffleType == CometNativeShuffle) {
      val (childPlan, childMetrics) = childPlanInfo
      val dep = CometShuffleExchangeExec.prepareShuffleDependency(
        inputRDD.asInstanceOf[RDD[ColumnarBatch]],
        child.output,
        outputPartitioning,
        serializer,
        metrics,
        childPlan,
        childMetrics)
      metrics("numPartitions").set(dep.partitioner.numPartitions)
      val executionId = sparkContext.getLocalProperty(SQLExecution.EXECUTION_ID_KEY)
      SQLMetrics.postDriverMetricUpdates(
        sparkContext,
        executionId,
        metrics("numPartitions") :: Nil)
      dep
    } else if (shuffleType == CometColumnarShuffle) {
```

Note: Also update the external call site in `CometTakeOrderedAndProjectExec` if it calls `prepareShuffleDependency`. Check by searching for calls to this method.

- [ ] **Step 7: Check for other call sites of prepareShuffleDependency**

Search for other callers. `CometTakeOrderedAndProjectExec` and `CometCollectLimitExec` also call `prepareShuffleDependency` — since we added default values `= None`, all existing callers continue to work without changes.

- [ ] **Step 8: Build to verify compilation**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw compile -pl spark -q`
Expected: BUILD SUCCESS

- [ ] **Step 9: Commit**

```bash
git add spark/src/main/scala/org/apache/spark/sql/comet/execution/shuffle/CometShuffleExchangeExec.scala
git commit -m "feat: detect eligible native children and fold into shuffle writer plan"
```

---

### Task 6: Run existing shuffle tests to verify no regressions

**Files:** (no changes — verification only)

- [ ] **Step 1: Build native code**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && make core`
Expected: Successful Rust build

- [ ] **Step 2: Run native shuffle tests**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -DwildcardSuites="CometNativeShuffleSuite" -q`
Expected: All tests pass. These tests exercise HashPartitioning, SinglePartition, and RoundRobinPartitioning with native shuffle, which are the eligible cases for the optimization.

- [ ] **Step 3: Run columnar shuffle tests to verify no impact**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -DwildcardSuites="CometColumnarShuffleSuite" -q`
Expected: All tests pass (columnar shuffle path is completely unaffected).

- [ ] **Step 4: Fix any failures and re-run**

If tests fail, debug the combined plan path. Common issues:
- Metrics tree depth mismatch: check that `CometMetricNode.fromCometPlan(nativeChild)` produces the right tree depth
- Schema mismatch: verify that `outputAttributes` in `getNativePlan()` still refers to the shuffle output schema (used for partitioning expressions), not the raw input schema
- Plan structure: add logging in `CometNativeShuffleWriter.getNativePlan()` to inspect the combined protobuf plan

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: address test failures in combined shuffle write path"
```

---

### Task 7: Add targeted test for the combined plan path

**Files:**
- Modify: `spark/src/test/scala/org/apache/comet/exec/CometNativeShuffleSuite.scala`

- [ ] **Step 1: Write test that verifies combined plan is used**

Add import at the top of the test file (if not already present):
```scala
import org.apache.spark.sql.comet.execution.shuffle.CometShuffleDependency
```

Add to `CometNativeShuffleSuite`:

```scala
  test("native shuffle: combined plan eliminates FFI round-trip") {
    // When a native child plan feeds into native shuffle write with a single input,
    // the child plan should be folded into the shuffle writer's native plan.
    withParquetTable((0 until 100).map(i => (i, (i + 1).toLong)), "tbl") {
      val df = sql("SELECT _1, _2 FROM tbl WHERE _1 > 10").repartition(10, col("_1"))

      // Verify correctness
      checkSparkAnswerAndOperator(df)

      // Verify the shuffle exchange uses native shuffle
      checkCometExchange(df, 1, true)

      // Verify the optimization is active: the CometShuffleDependency should have
      // a childNativePlan set
      val shuffleExchanges = collect(df.queryExecution.executedPlan) {
        case s: CometShuffleExchangeExec => s
      }
      assert(shuffleExchanges.nonEmpty, "Expected at least one CometShuffleExchangeExec")
      val dep = shuffleExchanges.head.shuffleDependency
        .asInstanceOf[CometShuffleDependency[_, _, _]]
      assert(dep.childNativePlan.isDefined,
        "Expected childNativePlan to be set (combined plan optimization active)")
    }
  }

  test("native shuffle: combined plan disabled by config") {
    withSQLConf(
      CometConf.COMET_EXEC_SHUFFLE_COMBINE_PLANS_ENABLED.key -> "false") {
      withParquetTable((0 until 100).map(i => (i, (i + 1).toLong)), "tbl") {
        val df = sql("SELECT _1, _2 FROM tbl WHERE _1 > 10").repartition(10, col("_1"))
        // Still produces correct results even with optimization disabled
        checkSparkAnswerAndOperator(df)
        checkCometExchange(df, 1, true)

        // Verify the optimization is NOT active
        val shuffleExchanges = collect(df.queryExecution.executedPlan) {
          case s: CometShuffleExchangeExec => s
        }
        assert(shuffleExchanges.nonEmpty)
        val dep = shuffleExchanges.head.shuffleDependency
          .asInstanceOf[CometShuffleDependency[_, _, _]]
        assert(dep.childNativePlan.isEmpty,
          "Expected childNativePlan to be empty (optimization disabled)")
      }
    }
  }

  test("native shuffle: fallback for RangePartitioning") {
    // RangePartitioning requires sampling the input RDD, so the optimization
    // should fall back to the current double-FFI path.
    withParquetTable((0 until 100).map(i => (i, (i + 1).toLong)), "tbl") {
      val df = sql("SELECT * FROM tbl").orderBy("_1")
      // orderBy triggers RangePartitioning shuffle
      checkSparkAnswer(df)
    }
  }

  test("native shuffle: fallback for multi-input plans (join)") {
    // Joins have two input sources, so the optimization should fall back.
    withParquetTable((0 until 50).map(i => (i, s"a$i")), "tbl1") {
      withParquetTable((0 until 50).map(i => (i, s"b$i")), "tbl2") {
        val df = sql("SELECT * FROM tbl1 JOIN tbl2 ON tbl1._1 = tbl2._1")
        checkSparkAnswer(df)
      }
    }
  }
```

- [ ] **Step 2: Run the new tests**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -Dsuites="org.apache.comet.exec.CometNativeShuffleSuite combined" -q`
Expected: All 4 new tests pass (combined plan, disabled config, RangePartitioning fallback, join fallback)

- [ ] **Step 3: Run full native shuffle suite**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -DwildcardSuites="CometNativeShuffleSuite" -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add spark/src/test/scala/org/apache/comet/exec/CometNativeShuffleSuite.scala
git commit -m "test: add tests for combined shuffle write plan path"
```

---

### Task 8: Run format and clippy checks

**Files:** (no changes — formatting only)

- [ ] **Step 1: Format all code**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && make format`
Expected: Code reformatted (Scala via scalafix + spotless, Rust via cargo fmt)

- [ ] **Step 2: Run clippy**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet/native && cargo clippy --all-targets --workspace -- -D warnings`
Expected: No warnings (no Rust changes were made, but verify)

- [ ] **Step 3: Commit any formatting changes**

```bash
git add -u
git commit -m "style: format code"
```

---

### Task 9: Final verification — full build and test

- [ ] **Step 1: Full build**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && make`
Expected: BUILD SUCCESS

- [ ] **Step 2: Run the broader exec test suite**

Run: `cd /Users/andy/git/apache/temp/datafusion-comet && ./mvnw test -DwildcardSuites="CometExec" -q`
Expected: All tests pass. This catches any regressions from queries that involve shuffle + native operators.

- [ ] **Step 3: Address any failures**

If tests fail, check whether the failure is in a query that hits the combined path. Toggle the config flag off (`spark.comet.exec.shuffle.combinePlans.enabled=false`) to confirm whether the failure is related to this change.
