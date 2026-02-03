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

# Code Coverage

Comet uses [JaCoCo](https://www.jacoco.org/jacoco/) for JVM code coverage reporting. Coverage reports
are generated locally when running tests.

## Generating Coverage Reports

Coverage reports are generated automatically when running tests with Maven:

```shell
./mvnw clean verify
```

Or via the Makefile:

```shell
make test-jvm
```

## Viewing Reports

After running tests, HTML coverage reports are available for each module:

- **common module**: `common/target/site/jacoco/index.html`
- **spark module**: `spark/target/site/jacoco/index.html`

Open these files in a web browser to view detailed coverage information including:

- Line coverage
- Branch coverage
- Coverage by package and class
- Source code with coverage highlighting

## Report Formats

JaCoCo generates reports in multiple formats:

| Format | Location | Use Case |
|--------|----------|----------|
| HTML | `target/site/jacoco/index.html` | Human-readable browsing |
| XML | `target/site/jacoco/jacoco.xml` | CI/CD tool integration |
| CSV | `target/site/jacoco/jacoco.csv` | Spreadsheet analysis |

## Running Tests for a Specific Module

To generate coverage for a specific module only:

```shell
./mvnw -pl common clean verify
./mvnw -pl spark clean verify
```

## Coverage for Specific Test Suites

To run a subset of tests and generate coverage:

```shell
MAVEN_OPTS="-DwildcardSuites=org.apache.comet.CometExpressionSuite" ./mvnw clean verify
```
