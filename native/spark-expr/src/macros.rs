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

/// Generates `PartialEq`, `Eq`, and `Hash` for an expression struct.
///
/// `#[derive(PartialEq, Hash)]` cannot be used on structs that hold
/// `Arc<dyn PhysicalExpr>` fields: because `dyn PhysicalExpr` is `?Sized`, the
/// derived `==` fails to compile with `E0507` ("cannot move out of a shared
/// reference"). This is the long-standing rustc limitation tracked at
/// <https://github.com/rust-lang/rust/issues/78808>, and upstream DataFusion
/// works around it the same way (see `BinaryExpr`). This macro generates the
/// same field-by-field impls by hand.
///
/// List exactly the fields that participate in equality and hashing. Fields
/// omitted from the list are ignored for both `eq` and `hash`: use this for
/// caches, `registry` / `query_context` handles, or fields whose type is not
/// `Eq` (such as a compiled `Regex`).
///
/// Equality uses `.eq()` rather than `==` so that it resolves through deref
/// coercion to `<dyn PhysicalExpr>::eq`; `==` would not compile for the
/// trait-object fields. This works unchanged for `Sized` fields too.
///
/// ```ignore
/// #[derive(Debug)]
/// pub struct NegativeExpr {
///     arg: Arc<dyn PhysicalExpr>,
///     fail_on_error: bool,
/// }
/// impl_expr_eq_hash!(NegativeExpr { arg, fail_on_error });
/// ```
#[macro_export]
macro_rules! impl_expr_eq_hash {
    ($ty:ty { $($field:ident),+ $(,)? }) => {
        impl ::std::cmp::PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                true $(&& self.$field.eq(&other.$field))+
            }
        }
        impl ::std::cmp::Eq for $ty {}
        impl ::std::hash::Hash for $ty {
            fn hash<H: ::std::hash::Hasher>(&self, state: &mut H) {
                $(self.$field.hash(state);)+
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_expr::PhysicalExpr;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    // A struct holding an `Arc<dyn PhysicalExpr>` field, which `#[derive]` cannot
    // produce `PartialEq`/`Hash` for (rust-lang/rust#78808). The macro must.
    #[derive(Debug)]
    struct DummyExpr {
        child: Arc<dyn PhysicalExpr>,
        ordinal: usize,
    }
    impl_expr_eq_hash!(DummyExpr { child, ordinal });

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    #[test]
    fn equal_instances_are_equal_and_hash_equally() {
        let a = DummyExpr {
            child: col("a", 0),
            ordinal: 1,
        };
        let b = DummyExpr {
            child: col("a", 0),
            ordinal: 1,
        };
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn differing_in_any_listed_field_is_unequal() {
        let base = DummyExpr {
            child: col("a", 0),
            ordinal: 1,
        };
        // Differs in the trait-object field.
        let diff_child = DummyExpr {
            child: col("b", 1),
            ordinal: 1,
        };
        // Differs in the scalar field.
        let diff_ordinal = DummyExpr {
            child: col("a", 0),
            ordinal: 2,
        };
        assert_ne!(base, diff_child);
        assert_ne!(base, diff_ordinal);
    }

    // Sanity check that the macro output satisfies trait bounds requiring `Eq + Hash`.
    #[test]
    fn satisfies_eq_hash_bounds() {
        fn requires_eq_hash<T: Eq + Hash>(_: &T) {}
        requires_eq_hash(&DummyExpr {
            child: col("a", 0),
            ordinal: 0,
        });
    }
}
