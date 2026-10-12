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

package io.github.jbellis.jvector.example.benchmarks.datasets;

import java.util.Locale;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;

/** Base-vector storage policy for benchmark harnesses, independent of quantization. */
public enum BaseVectorLoading {
    BUFFERED,
    PRELOAD;

    /** Existing VM option is an explicit override of the YAML flag. */
    public static BaseVectorLoading forBenchmark(boolean useDatasetPrefetching) {
        String value = System.getProperty("jvector.dataset_loader",
                useDatasetPrefetching ? "buffered" : "preload");
        try { return valueOf(value.toUpperCase(Locale.ROOT)); }
        catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("jvector.dataset_loader must be buffered or preload, got: " + value, e);
        }
    }

    public static BaseVectorLoading forBenchmark() { return forBenchmark(false); }

    /// Choose base-vector loading for one benchmark configuration.
    ///
    /// An explicitly set `jvector.dataset_loader` VM option is returned as-is: it overrides both
    /// the YAML `useDatasetPrefetching` flag and the routing below. Otherwise, opting in with the
    /// flag selects buffered input only where the harness consumes quantized or stored vectors;
    /// uncompressed construction and raw-input exact search stay on resident input.
    public static BaseVectorLoading forBenchmark(MultiConfig config) {
        var loading = forBenchmark(config.useDatasetPrefetching);
        if (System.getProperty("jvector.dataset_loader") != null) return loading;
        if (loading == PRELOAD) return loading;
        // Uncompressed construction still uses main's resident raw-vector source.
        if (config.construction.compression.stream().anyMatch(c -> "None".equals(c.type)))
            return PRELOAD;
        boolean exactSearch = config.search.compression.stream().anyMatch(c -> "None".equals(c.type));
        boolean needsRawInput = exactSearch && config.construction.getFeatureSets().stream().anyMatch(features ->
                !features.contains(FeatureId.FUSED_PQ));
        return needsRawInput ? PRELOAD : BUFFERED;
    }
}
