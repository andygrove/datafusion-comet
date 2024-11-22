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

use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) struct SparkPlan {
    /// Spark plan ID
    plan_id: u32,
    /// Root native plan that represents the Spark plan
    wrapped: Arc<dyn ExecutionPlan>,
    /// Additional native plans that contribute to metrics for the original Spark plan
    additional_plans: Vec<Arc<dyn ExecutionPlan>>,
}

impl SparkPlan {
    pub(crate) fn new(plan_id: u32, wrapped: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            plan_id,
            wrapped,
            additional_plans: vec![],
        }
    }

    pub(crate) fn new_with_additional(
        plan_id: u32,
        wrapped: Arc<dyn ExecutionPlan>,
        additional_plans: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self {
            plan_id,
            wrapped,
            additional_plans,
        }
    }
}

impl DisplayAs for SparkPlan {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        self.wrapped.fmt_as(t, f)?;
        write!(f, " (#{})", self.plan_id)
    }
}

impl ExecutionPlan for SparkPlan {
    fn name(&self) -> &str {
        self.wrapped.name()
    }

    fn as_any(&self) -> &dyn Any {
        self.wrapped.as_any()
    }

    fn properties(&self) -> &PlanProperties {
        self.wrapped.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.wrapped.children()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        unimplemented!()
        // Ok(Arc::new(SparkPlan {
        //     plan_id: self.plan_id.clone(),
        //     wrapped: self.wrapped.with_new_children(children)?,
        //     additional_plans: self.additional_plans.clone(),
        // }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        self.wrapped.execute(partition, context)
    }
}
