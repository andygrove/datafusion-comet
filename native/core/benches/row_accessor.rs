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

//! Benchmarks for SparkUnsafeRow field accessor methods.

use comet::execution::shuffle::row::{SparkUnsafeObject, SparkUnsafeRow};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

const NUM_ROWS: usize = 10000;
const NUM_COLS: usize = 10;

/// Row data layout for Spark UnsafeRow:
/// - First: null bitset (8 bytes per 64 columns)
/// - Then: 8 bytes per column (fixed-width values or offset+length for variable)
fn get_row_size(num_cols: usize) -> usize {
    SparkUnsafeRow::get_row_bitset_width(num_cols) + num_cols * 8
}

/// Creates a row buffer with test data.
/// Layout: [null_bitset][col0][col1]...[colN]
/// Each column slot is 8 bytes.
fn create_row_data(num_cols: usize) -> Vec<u8> {
    let row_size = get_row_size(num_cols);
    let mut data = vec![0u8; row_size];
    let bitset_width = SparkUnsafeRow::get_row_bitset_width(num_cols);

    // Set all columns as not null (null bits = 0)
    // Bitset is already 0

    // Write test values for each column
    for col in 0..num_cols {
        let offset = bitset_width + col * 8;
        // Write a known value: column index as i64
        let value = (col as i64 + 1) * 1000;
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    data
}

/// Creates multiple rows for batch benchmarking
fn create_rows(num_rows: usize, num_cols: usize) -> (Vec<Vec<u8>>, Vec<SparkUnsafeRow>) {
    let row_buffers: Vec<Vec<u8>> = (0..num_rows).map(|_| create_row_data(num_cols)).collect();

    let mut rows: Vec<SparkUnsafeRow> = (0..num_rows)
        .map(|_| SparkUnsafeRow::new_with_num_fields(num_cols))
        .collect();

    for (row, buffer) in rows.iter_mut().zip(row_buffers.iter()) {
        row.point_to_slice(buffer);
    }

    (row_buffers, rows)
}

fn bench_get_long(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_long");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    // Keep buffers alive for the duration of the benchmark
    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_long(0));
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("all_columns", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    sum = sum.wrapping_add(row.get_long(col));
                }
            }
            black_box(sum)
        });
    });

    drop(buffers); // Ensure buffers live until here
    group.finish();
}

fn bench_get_int(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_int");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i32 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_int(0));
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("all_columns", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i32 = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    sum = sum.wrapping_add(row.get_int(col));
                }
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_double(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_double");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: f64 = 0.0;
            for row in &rows {
                sum += row.get_double(0);
            }
            black_box(sum)
        });
    });

    group.bench_function(BenchmarkId::new("all_columns", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: f64 = 0.0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    sum += row.get_double(col);
                }
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_float(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_float");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: f32 = 0.0;
            for row in &rows {
                sum += row.get_float(0);
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_decimal(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_decimal");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    // Test with precision <= 18 (fast path - fits in i64)
    group.bench_function(BenchmarkId::new("precision_15", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i128 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_decimal(0, 15));
            }
            black_box(sum)
        });
    });

    // Test with precision <= 18, all columns
    group.bench_function(BenchmarkId::new("precision_15_all_cols", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i128 = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    sum = sum.wrapping_add(row.get_decimal(col, 15));
                }
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_date(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_date");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i32 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_date(0));
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_timestamp");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_timestamp(0));
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_boolean(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_boolean");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut count: usize = 0;
            for row in &rows {
                if row.get_boolean(0) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_byte(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_byte");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i32 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_byte(0) as i32);
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_get_short(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/get_short");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut sum: i32 = 0;
            for row in &rows {
                sum = sum.wrapping_add(row.get_short(0) as i32);
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn bench_is_null_at(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/is_null_at");
    group.throughput(Throughput::Elements(NUM_ROWS as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function(BenchmarkId::new("single_column", NUM_ROWS), |b| {
        b.iter(|| {
            let mut count: usize = 0;
            for row in &rows {
                if !row.is_null_at(0) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.bench_function(BenchmarkId::new("all_columns", NUM_ROWS), |b| {
        b.iter(|| {
            let mut count: usize = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    if !row.is_null_at(col) {
                        count += 1;
                    }
                }
            }
            black_box(count)
        });
    });

    drop(buffers);
    group.finish();
}

/// Combined benchmark simulating typical row processing:
/// check null, then read value for each column
fn bench_combined_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_accessor/combined");
    group.throughput(Throughput::Elements((NUM_ROWS * NUM_COLS) as u64));

    let (buffers, rows) = create_rows(NUM_ROWS, NUM_COLS);

    group.bench_function("null_check_and_get_long", |b| {
        b.iter(|| {
            let mut sum: i64 = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    if !row.is_null_at(col) {
                        sum = sum.wrapping_add(row.get_long(col));
                    }
                }
            }
            black_box(sum)
        });
    });

    group.bench_function("null_check_and_get_decimal_p15", |b| {
        b.iter(|| {
            let mut sum: i128 = 0;
            for row in &rows {
                for col in 0..NUM_COLS {
                    if !row.is_null_at(col) {
                        sum = sum.wrapping_add(row.get_decimal(col, 15));
                    }
                }
            }
            black_box(sum)
        });
    });

    drop(buffers);
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = config();
    targets =
        bench_get_long,
        bench_get_int,
        bench_get_double,
        bench_get_float,
        bench_get_decimal,
        bench_get_date,
        bench_get_timestamp,
        bench_get_boolean,
        bench_get_byte,
        bench_get_short,
        bench_is_null_at,
        bench_combined_access
}
criterion_main!(benches);
