

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

use std::{
    any::Any,
    fmt::{Debug, Display, Formatter},
    hash::{Hash, Hasher},
    sync::Arc,
};
use arrow::buffer::NullBuffer;
use arrow_array::{Array, BooleanArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, PhysicalExpr};
use datafusion_common::Result;
use crate::utils::down_cast_any_ref;

/// Specialization of `CASE WHEN predicate THEN expr ELSE null END`
#[derive(Debug, Hash)]
pub struct ExprOrNull {
    predicate: Arc<dyn PhysicalExpr>,
    expr: Arc<dyn PhysicalExpr>,
}

impl ExprOrNull {
    pub fn try_new(predicate: Arc<dyn PhysicalExpr>,
                   input: Arc<dyn PhysicalExpr>) -> Result<Self> {
        // TODO validation
        Ok(Self { predicate, expr: input })
    }
}

impl Display for ExprOrNull {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl DisplayAs for ExprOrNull {
    fn fmt_as(&self, _t: DisplayFormatType, _f: &mut Formatter) -> std::fmt::Result {
        todo!()
    }
}

impl PhysicalExpr for ExprOrNull {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        self.expr.data_type(input_schema)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        // evaluate predicate
        if let ColumnarValue::Array(bit_mask) = self.predicate.evaluate(batch)? {
            let bit_mask = bit_mask.as_any().downcast_ref::<BooleanArray>().expect("predicate should evaluate to a boolean array");
            // evaluate input expression
            if let ColumnarValue::Array(array ) = self.expr.evaluate(batch)? {
                let array = array.as_any().downcast_ref::<StringArray>().expect("input should be a primitive array");
                // create new array
                let (offsets, values, _) = array.to_owned().into_parts();
                let nulls = NullBuffer::new(bit_mask.values().to_owned());
                Ok(ColumnarValue::Array(Arc::new(StringArray::new(offsets, values, Some(nulls)))))
            } else {
                panic!()
            }
        } else {
            panic!()
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        todo!()
    }

    fn with_new_children(self: Arc<Self>, _children: Vec<Arc<dyn PhysicalExpr>>) -> Result<Arc<dyn PhysicalExpr>> {
        todo!()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut s = state;
        self.predicate.hash(&mut s);
        self.expr.hash(&mut s);
        self.hash(&mut s);
    }
}

impl PartialEq<dyn Any> for ExprOrNull {
    fn eq(&self, other: &dyn Any) -> bool {
        down_cast_any_ref(other)
            .downcast_ref::<Self>()
            .map(|x| {
                self.predicate.eq(&x.predicate)
                    && self.expr.eq(&x.expr)
            })
            .unwrap_or(false)
    }
}
