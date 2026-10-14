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

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertTrue;

/// Repro for #755: `TestNVQEncodingAllocation` fails intermittently, by 1-2 ULP, depending on
/// the host's vector width and on JIT timing.
///
/// ## The bug
///
/// `TestNVQEncodingAllocation.preservesFrozenEncodingAndLossBits` compares the reductions of
/// `nvqLoss`, `nvqUniformLoss`, `nvqSquareL2Distance8bit` and `nvqCosine8bit` bit for bit against a
/// `Reference` subclass. Both sides end in `FloatVector.reduceLanes(ADD)`, whose order of
/// additions the Vector API leaves unspecified: the Java fallback (interpreter, C1) and the C2
/// intrinsic can round differently. The test calls the two sides alternately in one JVM, so which
/// of them is compiled, and by which tier, varies from run to run. On this x86 host the default
/// 512-bit species happens to agree across tiers, but with a 256-bit or 128-bit preferred species
/// the two sides disagree in most runs, while both agree under `-Xint`, under `-Xcomp`, and under
/// C1 alone. That is the 1-2 ULP `nvqLoss` failure the PR's own CI shows on avx512f (JDK 24), the
/// ARM NEON runner, and the bench build. The kernels are not wrong; the bitwise oracle is not a
/// stable requirement across JIT states.
///
/// ## The test
///
/// Runs the PR's test in six fresh JVMs, three with `-XX:MaxVectorSize=32` (a 256-bit preferred
/// species, as on AVX2 hosts) and three with `-XX:MaxVectorSize=16` (128-bit, as on NEON), and
/// requires every run to pass. At `d903d206` individual runs fail about 60-80% of the time at these
/// widths, so all six passing by chance is very unlikely.
class TestNVQEncodingBitsAcrossJitStates {
    @Test
    void passesInEveryFreshJvmAtEveryVectorWidth() throws Exception {
        var outcomes = new ArrayList<KernelTestProbe.Outcome>();
        for (String width : new String[] {"-XX:MaxVectorSize=32", "-XX:MaxVectorSize=16"}) {
            for (int run = 0; run < 3; run++) {
                outcomes.add(KernelTestProbe.run("nvqEncodingBits", width));
            }
        }
        assertTrue(outcomes.stream().allMatch(KernelTestProbe.Outcome::passedOrSkipped),
                "TestNVQEncodingAllocation outcomes (3 runs at 256-bit, then 3 at 128-bit): "
                        + KernelTestProbe.describe(outcomes));
    }
}
