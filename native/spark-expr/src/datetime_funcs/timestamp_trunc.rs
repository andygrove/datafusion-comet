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

use crate::utils::array_with_timezone;
use arrow::datatypes::{DataType, Schema, TimeUnit::Microsecond};
use arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, ScalarValue::Utf8};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;
use std::hash::Hash;
use std::{
    any::Any,
    fmt::{Debug, Display, Formatter},
    sync::Arc,
};

use crate::kernels::temporal::{timestamp_trunc_array_fmt_dyn, timestamp_trunc_dyn};

#[derive(Debug, Eq)]
pub struct TimestampTruncExpr {
    /// An array with DataType::Timestamp(TimeUnit::Microsecond, None)
    child: Arc<dyn PhysicalExpr>,
    /// Scalar UTF8 string matching the valid values in Spark SQL: https://spark.apache.org/docs/latest/api/sql/index.html#date_trunc
    format: Arc<dyn PhysicalExpr>,
    /// String containing a timezone name. The name must be found in the standard timezone
    /// database (https://en.wikipedia.org/wiki/List_of_tz_database_time_zones). The string is
    /// later parsed into a chrono::TimeZone.
    /// Timestamp arrays in this implementation are kept in arrays of UTC timestamps (in micros)
    /// along with a single value for the associated TimeZone. The timezone offset is applied
    /// just before any operations on the timestamp
    timezone: String,
}

impl Hash for TimestampTruncExpr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        self.format.hash(state);
        self.timezone.hash(state);
    }
}
impl PartialEq for TimestampTruncExpr {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child)
            && self.format.eq(&other.format)
            && self.timezone.eq(&other.timezone)
    }
}

impl TimestampTruncExpr {
    pub fn new(
        child: Arc<dyn PhysicalExpr>,
        format: Arc<dyn PhysicalExpr>,
        timezone: String,
    ) -> Self {
        TimestampTruncExpr {
            child,
            format,
            timezone,
        }
    }
}

impl Display for TimestampTruncExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimestampTrunc [child:{}, format:{}, timezone: {}]",
            self.child, self.format, self.timezone
        )
    }
}

impl PhysicalExpr for TimestampTruncExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_sql(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }

    fn data_type(&self, input_schema: &Schema) -> datafusion::common::Result<DataType> {
        match self.child.data_type(input_schema)? {
            DataType::Dictionary(key_type, _) => Ok(DataType::Dictionary(
                key_type,
                Box::new(DataType::Timestamp(
                    Microsecond,
                    Some(self.timezone.clone().into()),
                )),
            )),
            _ => Ok(DataType::Timestamp(
                Microsecond,
                Some(self.timezone.clone().into()),
            )),
        }
    }

    fn nullable(&self, _: &Schema) -> datafusion::common::Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> datafusion::common::Result<ColumnarValue> {
        let timestamp = self.child.evaluate(batch)?;
        let format = self.format.evaluate(batch)?;
        let tz = self.timezone.clone();
        match (timestamp, format) {
            (ColumnarValue::Array(ts), ColumnarValue::Scalar(Utf8(Some(format)))) => {
                let ts = array_with_timezone(
                    ts,
                    tz.clone(),
                    Some(&DataType::Timestamp(Microsecond, Some(tz.clone().into()))),
                )?;
                let result = timestamp_trunc_dyn(&ts, format)?;
                // Add timezone to the result
                let result_with_tz = array_with_timezone(
                    result,
                    tz.clone(),
                    Some(&DataType::Timestamp(Microsecond, Some(tz.into()))),
                )?;
                Ok(ColumnarValue::Array(result_with_tz))
            }
            (ColumnarValue::Array(ts), ColumnarValue::Array(formats)) => {
                let ts = array_with_timezone(
                    ts,
                    tz.clone(),
                    Some(&DataType::Timestamp(Microsecond, Some(tz.clone().into()))),
                )?;
                let result = timestamp_trunc_array_fmt_dyn(&ts, &formats)?;
                // Add timezone to the result
                let result_with_tz = array_with_timezone(
                    result,
                    tz.clone(),
                    Some(&DataType::Timestamp(Microsecond, Some(tz.into()))),
                )?;
                Ok(ColumnarValue::Array(result_with_tz))
            }
            _ => Err(DataFusionError::Execution(
                "Invalid input to function TimestampTrunc. \
                    Expected (PrimitiveArray<TimestampMicrosecondType>, Scalar, String) or \
                    (PrimitiveArray<TimestampMicrosecondType>, StringArray, String)"
                    .to_string(),
            )),
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>, DataFusionError> {
        Ok(Arc::new(TimestampTruncExpr::new(
            Arc::clone(&children[0]),
            Arc::clone(&self.format),
            self.timezone.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, TimestampMicrosecondArray};
    use arrow::datatypes::{Field, Schema as ArrowSchema};
    use std::sync::Arc;

    #[test]
    fn test_timestamp_trunc_with_timezone_utc_to_denver() {
        // Create a timestamp array with UTC timezone
        // Using a specific timestamp: 2024-01-15 10:30:45 UTC
        let timestamp_micros = 1705318245000000i64; // 2024-01-15 10:30:45 UTC
        let ts_array = TimestampMicrosecondArray::from(vec![Some(timestamp_micros)])
            .with_timezone("UTC");
        let ts_array_ref = Arc::new(ts_array) as ArrayRef;

        // Create schema with UTC timestamp
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(Microsecond, Some("UTC".into())),
            true,
        )]));

        // Create RecordBatch
        let batch = RecordBatch::try_new(schema.clone(), vec![ts_array_ref]).unwrap();

        // Create column expression (references column 0)
        let child_expr = Arc::new(datafusion::physical_plan::expressions::Column::new("ts", 0));

        // Create format expression ("HOUR")
        let format_expr = Arc::new(datafusion::physical_plan::expressions::Literal::new(
            Utf8(Some("HOUR".to_string())),
        ));

        // Create TimestampTruncExpr with America/Denver timezone
        let trunc_expr = TimestampTruncExpr::new(
            child_expr as Arc<dyn PhysicalExpr>,
            format_expr as Arc<dyn PhysicalExpr>,
            "America/Denver".to_string(),
        );

        // Evaluate
        let result = trunc_expr.evaluate(&batch);

        // Print result for debugging
        match &result {
            Ok(ColumnarValue::Array(arr)) => {
                println!("Result array data type: {:?}", arr.data_type());
                println!("Result array values: {:?}", arr);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
            _ => {}
        }

        // Check if it succeeded or got an error
        assert!(
            result.is_ok(),
            "Expected successful evaluation but got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_timestamp_trunc_with_matching_timezone() {
        // Create a timestamp array with America/Denver timezone
        let timestamp_micros = 1705318245000000i64;
        let ts_array = TimestampMicrosecondArray::from(vec![Some(timestamp_micros)])
            .with_timezone("America/Denver");
        let ts_array_ref = Arc::new(ts_array) as ArrayRef;

        // Create schema with America/Denver timestamp
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(Microsecond, Some("America/Denver".into())),
            true,
        )]));

        // Create RecordBatch
        let batch = RecordBatch::try_new(schema.clone(), vec![ts_array_ref]).unwrap();

        // Create column expression
        let child_expr = Arc::new(datafusion::physical_plan::expressions::Column::new("ts", 0));

        // Create format expression ("HOUR")
        let format_expr = Arc::new(datafusion::physical_plan::expressions::Literal::new(
            Utf8(Some("HOUR".to_string())),
        ));

        // Create TimestampTruncExpr with matching America/Denver timezone
        let trunc_expr = TimestampTruncExpr::new(
            child_expr as Arc<dyn PhysicalExpr>,
            format_expr as Arc<dyn PhysicalExpr>,
            "America/Denver".to_string(),
        );

        // Evaluate
        let result = trunc_expr.evaluate(&batch);

        // Print result for debugging
        match &result {
            Ok(ColumnarValue::Array(arr)) => {
                println!("Result array data type: {:?}", arr.data_type());
                println!("Result array values: {:?}", arr);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
            _ => {}
        }

        assert!(
            result.is_ok(),
            "Expected successful evaluation but got error: {:?}",
            result.err()
        );
    }
}
