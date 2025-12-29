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

use arrow::array::{ArrayRef, TimestampMicrosecondArray};
use arrow::datatypes::Field;
use arrow::error::Result as ArrowResult;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion::common::config::ConfigOptions;
use datafusion::logical_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_comet_spark_expr::{SparkHour, SparkMinute, SparkSecond};
use std::sync::Arc;

fn create_timestamp_array(size: usize) -> ArrayRef {
    let mut values = Vec::with_capacity(size);
    let base_timestamp = 1640000000000000i64; // 2021-12-20 11:33:20 UTC
    for i in 0..size {
        if i % 10 == 0 {
            values.push(None);
        } else {
            // Add varying microseconds to create different timestamps
            values.push(Some(base_timestamp + (i as i64) * 1000000));
        }
    }
    let array = TimestampMicrosecondArray::from(values).with_timezone("America/Los_Angeles");
    Arc::new(array)
}

fn invoke_udf(udf_impl: &dyn ScalarUDFImpl, array: ArrayRef) -> ArrowResult<ColumnarValue> {
    let return_type = udf_impl.return_type(&[array.data_type().clone()]).unwrap();
    let num_rows = array.len();
    let args = ScalarFunctionArgs {
        args: vec![ColumnarValue::Array(array)],
        number_rows: num_rows,
        arg_fields: vec![],
        return_field: Arc::new(Field::new("result", return_type, true)),
        config_options: Arc::new(ConfigOptions::default()),
    };
    udf_impl
        .invoke_with_args(args)
        .map_err(|e| arrow::error::ArrowError::ComputeError(format!("DataFusion error: {}", e)))
}

fn benchmark_extract_hour(c: &mut Criterion) {
    let array = create_timestamp_array(1000);
    let hour_impl = SparkHour::new("America/Los_Angeles".to_string());

    c.bench_function("extract_hour_1000", |b| {
        b.iter(|| invoke_udf(&hour_impl, array.clone()).unwrap());
    });
}

fn benchmark_extract_minute(c: &mut Criterion) {
    let array = create_timestamp_array(1000);
    let minute_impl = SparkMinute::new("America/Los_Angeles".to_string());

    c.bench_function("extract_minute_1000", |b| {
        b.iter(|| invoke_udf(&minute_impl, array.clone()).unwrap());
    });
}

fn benchmark_extract_second(c: &mut Criterion) {
    let array = create_timestamp_array(1000);
    let second_impl = SparkSecond::new("America/Los_Angeles".to_string());

    c.bench_function("extract_second_1000", |b| {
        b.iter(|| invoke_udf(&second_impl, array.clone()).unwrap());
    });
}

fn benchmark_all_sizes(c: &mut Criterion) {
    let hour_impl = SparkHour::new("America/Los_Angeles".to_string());

    let mut group = c.benchmark_group("extract_hour_varying_sizes");
    for size in [100, 1000, 10000] {
        let array = create_timestamp_array(size);
        group.bench_function(format!("size_{}", size), |b| {
            b.iter(|| invoke_udf(&hour_impl, array.clone()).unwrap());
        });
    }
    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = config();
    targets = benchmark_extract_hour, benchmark_extract_minute, benchmark_extract_second, benchmark_all_sizes
}
criterion_main!(benches);
