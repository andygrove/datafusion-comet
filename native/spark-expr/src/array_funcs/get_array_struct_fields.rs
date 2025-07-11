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

use arrow::array::{make_array, Array, GenericListArray, OffsetSizeTrait, StructArray};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::common::{
    cast::{as_large_list_array, as_list_array},
    internal_err, DataFusionError, Result as DataFusionResult,
};
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::PhysicalExpr;
use std::hash::Hash;
use std::{
    any::Any,
    fmt::{Debug, Display, Formatter},
    sync::Arc,
};

#[derive(Debug, Eq)]
pub struct GetArrayStructFields {
    child: Arc<dyn PhysicalExpr>,
    ordinal: usize,
}

impl Hash for GetArrayStructFields {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        self.ordinal.hash(state);
    }
}
impl PartialEq for GetArrayStructFields {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.ordinal.eq(&other.ordinal)
    }
}

impl GetArrayStructFields {
    pub fn new(child: Arc<dyn PhysicalExpr>, ordinal: usize) -> Self {
        Self { child, ordinal }
    }

    fn list_field(&self, input_schema: &Schema) -> DataFusionResult<FieldRef> {
        match self.child.data_type(input_schema)? {
            DataType::List(field) | DataType::LargeList(field) => Ok(field),
            data_type => Err(DataFusionError::Internal(format!(
                "Unexpected data type in GetArrayStructFields: {data_type:?}"
            ))),
        }
    }

    fn child_field(&self, input_schema: &Schema) -> DataFusionResult<FieldRef> {
        match self.list_field(input_schema)?.data_type() {
            DataType::Struct(fields) => Ok(Arc::clone(&fields[self.ordinal])),
            data_type => Err(DataFusionError::Internal(format!(
                "Unexpected data type in GetArrayStructFields: {data_type:?}"
            ))),
        }
    }
}

impl PhysicalExpr for GetArrayStructFields {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> DataFusionResult<DataType> {
        let struct_field = self.child_field(input_schema)?;
        match self.child.data_type(input_schema)? {
            DataType::List(_) => Ok(DataType::List(struct_field)),
            DataType::LargeList(_) => Ok(DataType::LargeList(struct_field)),
            data_type => Err(DataFusionError::Internal(format!(
                "Unexpected data type in GetArrayStructFields: {data_type:?}"
            ))),
        }
    }

    fn nullable(&self, input_schema: &Schema) -> DataFusionResult<bool> {
        Ok(self.list_field(input_schema)?.is_nullable()
            || self.child_field(input_schema)?.is_nullable())
    }

    fn evaluate(&self, batch: &RecordBatch) -> DataFusionResult<ColumnarValue> {
        let child_value = self.child.evaluate(batch)?.into_array(batch.num_rows())?;

        match child_value.data_type() {
            DataType::List(_) => {
                let list_array = as_list_array(&child_value)?;

                get_array_struct_fields(list_array, self.ordinal)
            }
            DataType::LargeList(_) => {
                let list_array = as_large_list_array(&child_value)?;

                get_array_struct_fields(list_array, self.ordinal)
            }
            data_type => Err(DataFusionError::Internal(format!(
                "Unexpected child type for ListExtract: {data_type:?}"
            ))),
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.child]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> datafusion::common::Result<Arc<dyn PhysicalExpr>> {
        match children.len() {
            1 => Ok(Arc::new(GetArrayStructFields::new(
                Arc::clone(&children[0]),
                self.ordinal,
            ))),
            _ => internal_err!("GetArrayStructFields should have exactly one child"),
        }
    }

    fn fmt_sql(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

fn get_array_struct_fields<O: OffsetSizeTrait>(
    list_array: &GenericListArray<O>,
    ordinal: usize,
) -> DataFusionResult<ColumnarValue> {
    let values = list_array
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("A StructType is expected");

    let field = Arc::clone(&values.fields()[ordinal]);
    // Get struct column by ordinal
    let extracted_column = values.column(ordinal);

    let data = if values.null_count() == extracted_column.null_count() {
        Arc::clone(extracted_column)
    } else {
        // In some cases the column obtained from struct by ordinal doesn't
        // represent all nulls that imposed by parent values.
        // This maybe caused by a low level reader bug and needs more investigation.
        // For this specific case we patch the null buffer for the column by merging nulls buffers
        // from parent and column
        let merged_nulls = NullBuffer::union(values.nulls(), extracted_column.nulls());
        make_array(
            extracted_column
                .into_data()
                .into_builder()
                .nulls(merged_nulls)
                .build()?,
        )
    };

    let array = GenericListArray::new(
        field,
        list_array.offsets().clone(),
        data,
        list_array.nulls().cloned(),
    );

    Ok(ColumnarValue::Array(Arc::new(array)))
}

impl Display for GetArrayStructFields {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GetArrayStructFields [child: {:?}, ordinal: {:?}]",
            self.child, self.ordinal
        )
    }
}
