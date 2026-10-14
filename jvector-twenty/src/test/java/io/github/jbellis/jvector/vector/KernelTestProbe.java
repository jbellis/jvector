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

import org.opentest4j.TestAbortedException;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

/// Runs one of this package's kernel tests in a fresh JVM with extra JVM options, so a test can
/// be checked under a host shape this machine doesn't have (e.g. `-XX:MaxVectorSize=16` gives a
/// 128-bit preferred species, as on ARM NEON) and with a fresh JIT state each time.
///
/// The child prints `PASS`, `SKIP` (the test aborted on an assumption) or `FAIL <message>`.
final class KernelTestProbe {
    private KernelTestProbe() {}

    /// Result of one child run: its outcome line and full output.
    record Outcome(String result, String output) {
        boolean passedOrSkipped() { return result.equals("PASS") || result.startsWith("SKIP"); }
    }

    static Outcome run(String test, String... jvmOptions) throws Exception {
        var command = new ArrayList<String>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.add("--add-modules=jdk.incubator.vector");
        command.addAll(Arrays.asList(jvmOptions));
        command.add("-cp");
        command.add(System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
        command.add(Child.class.getName());
        command.add(test);
        var process = new ProcessBuilder(command).redirectErrorStream(true).start();
        if (!process.waitFor(120, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new AssertionError("probe timed out: " + test + " " + Arrays.toString(jvmOptions));
        }
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        String result = output.lines().filter(l -> l.startsWith("PASS") || l.startsWith("SKIP") || l.startsWith("FAIL"))
                .reduce((first, second) -> second).orElse("FAIL no result: " + output);
        return new Outcome(result, output);
    }

    /// Child entry point: runs the named test method and reports its outcome on stdout.
    static final class Child {
        public static void main(String[] args) {
            try {
                switch (args[0]) {
                    case "nvqEncodingBits" -> new TestNVQEncodingAllocation().preservesFrozenEncodingBitsAndLossValues();
                    case "dot256ExactWidth" -> new TestDotProduct256().exactWidthPreservesOriginalReductionBits();
                    default -> throw new IllegalArgumentException("unknown test " + args[0]);
                }
                System.out.println("PASS");
            } catch (TestAbortedException e) {
                System.out.println("SKIP " + e.getMessage());
            } catch (Throwable t) {
                System.out.println("FAIL " + t);
            }
        }
    }

    static List<String> describe(List<Outcome> outcomes) {
        return outcomes.stream().map(Outcome::result).toList();
    }
}
