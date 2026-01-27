# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Apache DataFusion Comet is a high-performance accelerator for Apache Spark, built on Apache DataFusion. It's a multi-language project with:
- Native code in Rust (under `native/` directory)
- JVM code in Java and Scala (under `common/`, `spark/`, and related directories)
- Protobuf for JVM-Rust communication

## Build Commands

### Full Build
- `make` - Compile the entire project without running tests
- `make release` - Create a release build and install to local Maven repository
- `make clean` - Clean all build artifacts

### Building Components
- `make core` or `cd native && cargo build` - Build Rust native libraries
- `make jvm` - Build JVM components (requires native build first)

### Running Tests
- `make test` - Run all tests (Rust and JVM)
- `make test-rust` - Run only Rust tests
- `make test-jvm` - Run only JVM tests (requires native build)
- `cd native && cargo test` - Run Rust tests directly

### Running a Single Test
```bash
# Run a specific ScalaTest suite (use -Dtest=none to skip JUnit tests)
./mvnw test -Pspark-3.5 -Dtest=none -Dsuites="org.apache.comet.CometCastSuite"

# Run a specific test within a suite
./mvnw test -Pspark-3.5 -Dtest=none -Dsuites="org.apache.comet.CometCastSuite valid"

# Run multiple suites matching a pattern (wildcards supported)
./mvnw test -Pspark-3.5 -Dtest=none -Dsuites="*Iceberg*"

# Run with specific scan implementation
COMET_NATIVE_SCAN_IMPL=native_datafusion ./mvnw test -Pspark-3.5 -Dtest=none -Dsuites="org.apache.comet.CometExpressionSuite"
```

**Important**: Do not use `-pl` flag with test commands - the Maven reactor discovers test modules automatically.

### Code Formatting and Linting
- `make format` - Format all code (Rust, Scala, Java)
- `cd native && cargo fmt` - Format Rust code
- `./mvnw compile test-compile scalafix:scalafix -Psemanticdb` - Run Scalafix
- `./mvnw spotless:apply` - Apply Spotless formatting
- `cd native && cargo clippy --color=never --all-targets --workspace -- -D warnings` - Run Clippy checks

### Benchmarking
- `make benchmark-<ClassName>` - Run specific benchmark
- `make bench` - Run Rust benchmarks

## Architecture

### Project Structure
- `native/` - Rust code implementing DataFusion-based query execution
  - `core/` - Core execution engine, JNI bridge, shuffle, operators
  - `spark-expr/` - Spark expression implementations in Rust
  - `proto/` - Protobuf definitions for JVM-Rust communication
- `common/` - Common Java/Scala code shared across modules
- `spark/` - Spark integration and Comet operators
- `docs/` - Documentation
- `dev/` - Development tools, diffs for Spark versions, benchmarking scripts

### Key Components
1. **CometScan** - Native Parquet reader implementations
   - `SCAN_NATIVE_COMET` - Default Comet scanner
   - `SCAN_NATIVE_DATAFUSION` - Pure DataFusion scanner
   - `SCAN_NATIVE_ICEBERG_COMPAT` - Iceberg-compatible scanner

2. **CometExec Operators** - Native implementations of Spark operators
   - CometProjectExec, CometFilterExec, CometSortExec, etc.
   - Communicate with Rust via JNI and protobuf

3. **Shuffle** - Native shuffle implementation
   - CometShuffleManager - Custom Spark shuffle manager
   - Native shuffle writer in Rust for better performance

4. **Expression Evaluation** - Native implementations of Spark expressions
   - String functions, math functions, date/time functions, etc.
   - Located in `native/spark-expr/src/`

## Development Workflow

### Before Opening in IDE
Run `make` once after cloning to generate protobuf classes.

### Testing Changes
1. Make code changes
2. Run `make core` if you changed Rust code
3. Run relevant tests
4. Run `make format` before committing

### Spark Version Support
- Spark 3.4, 3.5, 4.0 are supported
- Use Maven profiles: `-Pspark-3.4`, `-Pspark-3.5`, `-Pspark-4.0`
- Spark 4.0 requires JDK 17+

### Running Spark SQL Tests
Apply diffs from `dev/diffs/` to Spark source to run Spark SQL tests with Comet:
```bash
git apply ../datafusion-comet/dev/diffs/3.5.6.diff
ENABLE_COMET=true build/sbt "sql/testOnly *"
```

## Important Configuration

Key Comet configurations (set via Spark conf):
- `spark.comet.enabled` - Enable Comet (default: false)
- `spark.comet.exec.enabled` - Enable native operators
- `spark.comet.exec.shuffle.enabled` - Enable native shuffle
- `spark.comet.scan.impl` - Scanner implementation choice

## Debugging

- Set `RUST_BACKTRACE=1` for Rust stack traces
- Use `spark.comet.debug.enabled=true` for debug logging
- See `docs/source/contributor-guide/debugging.md` for concurrent JVM/Rust debugging