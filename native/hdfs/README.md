<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Apache DataFusion Comet: HDFS integration

This crate contains the HDFS cluster integration 
and is intended to be used as part of the Apache DataFusion Comet project

The HDFS access powered by [fs-hdfs](https://github.com/datafusion-contrib/fs-hdfs).
The crate provides `object_store` implementation leveraged by Rust FFI APIs for the `libhdfs` which can be compiled 
by a set of C files provided by the [official Hadoop Community](https://github.com/apache/hadoop).

# Supported HDFS versions

Currently supported Apache Hadoop clients are: 
- 2.* 
- 3.*