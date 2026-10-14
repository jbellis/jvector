/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.vector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/// Repro for #755: `TestDotProduct256.exactWidthPreservesOriginalReductionBits` fails on every
/// host whose preferred species is 128 bits, such as the ARM NEON CI runner.
///
/// ## The bug
///
/// The test's expected value is one 8-lane `SPECIES_256` multiply followed by `reduceLanes(ADD)`,
/// and it asserts that `dotProduct(a, offset, b, offset + 2, 8)` returns those exact bits. That
/// holds only where a length-8 dot product is routed to `dot256` (512-bit preferred species) or to
/// an 8-lane preferred kernel (256-bit). With a 4-lane preferred species, `dotProduct` sends length
/// 8 to `dotProductPreferred`, which adds two 4-lane products lane-wise before reducing, a
/// different association. That is the PR's ARM CI failure,
/// `expected: <-1088322672> but was: <-1088322673>`. The kernel is fine (main routes the same
/// way); the test asserts a property the 128-bit dispatch never had.
///
/// ## The test
///
/// Runs the PR's test in a fresh JVM with `-XX:MaxVectorSize=16`, which gives this x86 host a
/// 128-bit preferred species, and requires it to pass or to skip. At `d903d206` it fails with the
/// same bits as on ARM, on every run.
class TestDotProduct256OnNarrowHosts {
    @Test
    void exactWidthTestHoldsOnA128BitHost() throws Exception {
        var outcome = KernelTestProbe.run("dot256ExactWidth", "-XX:MaxVectorSize=16");
        assertTrue(outcome.passedOrSkipped(), "TestDotProduct256 with a 128-bit preferred species: " + outcome.result());
    }
}
