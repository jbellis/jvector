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

import org.junit.Assume;
import org.junit.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Fresh JVMs exercise the real singleton without leaking provider properties between tests. */
public class TestVectorizationProviderSelection {
    private static final String ENABLE_NATIVE = "-Djvector.enable_native_vectorization=";
    private static final String LEGACY_ENABLE_NATIVE = "-Djvector.experimental.enable_native_vectorization=";
    private static final boolean HAS_VECTOR_MODULE = VectorizationProvider.vectorModulePresentAndReadable();

    private static String classpath() {
        return System.getProperty("surefire.test.class.path", System.getProperty("java.class.path"));
    }

    private static String withoutNativeArtifact() {
        return Arrays.stream(classpath().split(java.util.regex.Pattern.quote(File.pathSeparator)))
                .filter(entry -> !entry.replace('\\', '/').contains("/jvector-native/"))
                .filter(entry -> !(Path.of(entry).getFileName().toString().startsWith("jvector-native-")
                        && entry.endsWith(".jar")))
                .collect(Collectors.joining(File.pathSeparator));
    }

    private static boolean nativeAvailable() throws Exception {
        if (Runtime.version().feature() < 22 || !HAS_VECTOR_MODULE
                || !System.getProperty("os.name").startsWith("Linux")) return false;
        String arch = System.getProperty("os.arch");
        String resource = arch.equals("aarch64") ? "/libjvector-aarch64.so"
                : (arch.equals("amd64") || arch.equals("x86_64")) ? "/libjvector-x86_64.so" : null;
        if (resource == null) return false;
        try {
            return Class.forName("io.github.jbellis.jvector.vector.NativeVectorizationProvider", false,
                    TestVectorizationProviderSelection.class.getClassLoader()).getResource(resource) != null;
        } catch (ClassNotFoundException e) { return false; }
    }

    private static String javaProvider() {
        return HAS_VECTOR_MODULE ? "PanamaVectorizationProvider" : "DefaultVectorizationProvider";
    }

    private static String probe(String cp, boolean vectorModule, boolean success, String... properties) throws Exception {
        return probe(cp, vectorModule, true, success, properties);
    }

    /// Run [Probe] in a fresh JVM. `nativeAccess` controls whether the child gets
    /// `--enable-native-access=ALL-UNNAMED` (Java 22+ only).
    private static String probe(String cp, boolean vectorModule, boolean nativeAccess, boolean success,
                                String... properties) throws Exception {
        var tmp = Files.createTempDirectory("provider-selection-");
        Process child = null;
        try {
            var command = new ArrayList<String>();
            command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
            if (vectorModule) command.add("--add-modules=jdk.incubator.vector");
            if (nativeAccess && Runtime.version().feature() >= 22) command.add("--enable-native-access=ALL-UNNAMED");
            command.add("-Djava.io.tmpdir=" + tmp);
            command.addAll(Arrays.asList(properties));
            command.add("-cp"); command.add(cp); command.add(Probe.class.getName());
            var process = new ProcessBuilder(command).redirectErrorStream(true).start();
            child = process;
            assertTrue("Provider probe timed out", process.waitFor(30, TimeUnit.SECONDS));
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (success) assertEquals(output, 0, process.exitValue());
            else assertTrue(output, process.exitValue() != 0);
            return output;
        } finally {
            if (child != null && child.isAlive()) { child.destroyForcibly(); child.waitFor(); }
            try (var files = Files.walk(tmp)) {
                for (var file : files.sorted(Comparator.reverseOrder()).collect(Collectors.toList()))
                    Files.delete(file);
            }
        }
    }

    private static void assertProvider(String expected, String cp, String... properties) throws Exception {
        String output = probe(cp, HAS_VECTOR_MODULE, true, properties);
        assertTrue(output, output.contains("PROVIDER=" + expected));
        // The native distance and inherited sum must both work with the selected vector storage.
        assertTrue(output, output.contains("DOT=14.0 SUM=6.0"));
    }

    @Test public void automaticSelectionPrefersAvailableNative() throws Exception {
        assertProvider(nativeAvailable() ? "NativeVectorizationProvider" : javaProvider(), classpath());
    }

    @Test public void explicitFalseDisablesAutomaticNative() throws Exception {
        assertProvider(javaProvider(), classpath(), ENABLE_NATIVE + "false");
    }

    @Test public void explicitTrueStillEnablesNative() throws Exception {
        assertProvider(nativeAvailable() ? "NativeVectorizationProvider" : javaProvider(), classpath(), ENABLE_NATIVE + "true");
    }

    @Test public void deprecatedFalseAliasStillDisablesNative() throws Exception {
        assertProvider(javaProvider(), classpath(), LEGACY_ENABLE_NATIVE + "false");
    }

    @Test public void deprecatedTrueAliasStillEnablesNative() throws Exception {
        assertProvider(nativeAvailable() ? "NativeVectorizationProvider" : javaProvider(), classpath(),
                LEGACY_ENABLE_NATIVE + "true");
    }

    @Test public void newFalseOverridesDeprecatedTrue() throws Exception {
        assertProvider(javaProvider(), classpath(), ENABLE_NATIVE + "false", LEGACY_ENABLE_NATIVE + "true");
    }

    @Test public void newTrueOverridesDeprecatedFalse() throws Exception {
        assertProvider(nativeAvailable() ? "NativeVectorizationProvider" : javaProvider(), classpath(),
                LEGACY_ENABLE_NATIVE + "false", ENABLE_NATIVE + "true");
    }

    @Test public void explicitScalarProviderTakesPrecedence() throws Exception {
        assertProvider("DefaultVectorizationProvider", classpath(), ENABLE_NATIVE + "true", LEGACY_ENABLE_NATIVE + "false",
                "-Djvector.vectorization_provider=default");
    }

    @Test public void explicitPanamaProviderTakesPrecedence() throws Exception {
        Assume.assumeTrue(HAS_VECTOR_MODULE);
        assertProvider("PanamaVectorizationProvider", classpath(), ENABLE_NATIVE + "true", LEGACY_ENABLE_NATIVE + "false",
                "-Djvector.vectorization_provider=panama");
    }

    @Test public void explicitNativeProviderTakesPrecedenceOverFalse() throws Exception {
        Assume.assumeTrue(nativeAvailable());
        assertProvider("NativeVectorizationProvider", classpath(), ENABLE_NATIVE + "false", LEGACY_ENABLE_NATIVE + "true",
                "-Djvector.vectorization_provider=native");
    }

    @Test public void absentNativeArtifactUsesJavaProvider() throws Exception {
        assertProvider(javaProvider(), withoutNativeArtifact());
    }

    @Test public void unsupportedArchitectureUsesJavaProvider() throws Exception {
        assertProvider(javaProvider(), classpath(), "-Dos.arch=unsupported-provider-test");
    }

    @Test public void explicitNativeFailureDoesNotSilentlyFallBack() throws Exception {
        String output = probe(classpath(), HAS_VECTOR_MODULE, false,
                "-Djvector.vectorization_provider=native", "-Dos.arch=unsupported-provider-test");
        assertTrue(output, output.contains("Failed to load forced NativeVectorizationProvider"));
    }

    @Test public void absentVectorModuleUsesScalar() throws Exception {
        String output = probe(classpath(), false, true);
        assertTrue(output, output.contains("PROVIDER=DefaultVectorizationProvider"));
    }

    /// Repro for #751: when native access is denied, provider selection fails instead of falling
    /// back to the Java provider.
    ///
    /// ## The bug
    ///
    /// Automatic selection now constructs `NativeVectorizationProvider` by default on Java 22+.
    /// Its constructor calls `LibraryLoader.loadJvector()`, whose first `try` wraps
    /// `System.loadLibrary("jvector")` and catches only `UnsatisfiedLinkError`. When native
    /// access is denied (`--illegal-native-access=deny` on Java 24+, and the announced future
    /// default), `loadLibrary` throws `IllegalCallerException`, which escapes the loader. The
    /// constructor's `InvocationTargetException` then has a cause other than
    /// `UnsupportedOperationException`, so `VectorizationProvider.lookup` throws
    /// `AssertionError`. The provider holder fails to initialize, and no vector operation in the
    /// JVM can run. Before #751 native was opt-in, so the same JVM used Panama, and still does
    /// with `-Djvector.enable_native_vectorization=false`.
    ///
    /// ## The test
    ///
    /// Runs the probe without `--enable-native-access` and with `--illegal-native-access=deny`,
    /// and expects the Java provider to be selected and to compute correctly. At `599478a1`
    /// the child exits with `AssertionError` caused by `IllegalCallerException`.
    @Test public void deniedNativeAccessFallsBackToJavaProvider() throws Exception {
        Assume.assumeTrue("--illegal-native-access needs Java 24+", Runtime.version().feature() >= 24);
        String output = probe(classpath(), HAS_VECTOR_MODULE, false, true, "--illegal-native-access=deny");
        assertTrue(output, output.contains("PROVIDER=" + javaProvider()));
        assertTrue(output, output.contains("DOT=14.0 SUM=6.0"));
    }

    public static class Probe {
        public static void main(String[] args) {
            var provider = VectorizationProvider.getInstance();
            var vector = provider.getVectorTypeSupport().createFloatVector(new float[] {1f, 2f, 3f});
            var support = provider.getVectorUtilSupport();
            System.out.println("PROVIDER=" + provider.getClass().getSimpleName());
            System.out.println("DOT=" + support.dotProduct(vector, vector) + " SUM=" + support.sum(vector));
        }
    }
}
