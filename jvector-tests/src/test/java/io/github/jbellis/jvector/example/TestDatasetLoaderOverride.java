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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.benchmarks.datasets.BaseVectorLoading;
import io.github.jbellis.jvector.example.yaml.ConstructionParameters;
import io.github.jbellis.jvector.example.yaml.MultiConfig;
import io.github.jbellis.jvector.example.yaml.SearchParameters;
import io.github.jbellis.jvector.example.yaml.Compression;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/// The docs, release note and PR description all describe `-Djvector.dataset_loader` as an
/// explicit override of the YAML `useDatasetPrefetching` flag. These cases set the property to
/// `buffered` on configs that the harness routing sends to resident input, and expect the
/// documented override to win.
public class TestDatasetLoaderOverride {
    private static final String PROPERTY = "jvector.dataset_loader";
    private String previous;

    @Before public void saveProperty() { previous = System.getProperty(PROPERTY); }

    @After public void restoreProperty() {
        if (previous == null) System.clearProperty(PROPERTY);
        else System.setProperty(PROPERTY, previous);
    }

    private static MultiConfig config(String build, String search) {
        var config = new MultiConfig();
        config.construction = new ConstructionParameters();
        config.search = new SearchParameters();
        var bc = new Compression(); bc.type = build;
        var sc = new Compression(); sc.type = search;
        config.construction.compression = List.of(bc);
        config.construction.fusedGraph = List.of(false);
        config.construction.reranking = List.of("NVQ");
        config.search.compression = List.of(sc);
        return config;
    }

    @Test public void explicitBufferedPropertyOverridesExactSearchRouting() {
        System.setProperty(PROPERTY, "buffered");
        assertEquals("explicit -Djvector.dataset_loader=buffered was not honored for search compression None",
                BaseVectorLoading.BUFFERED, BaseVectorLoading.forBenchmark(config("PQ", "None")));
    }

    @Test public void explicitBufferedPropertyOverridesUncompressedConstructionRouting() {
        System.setProperty(PROPERTY, "buffered");
        assertEquals("explicit -Djvector.dataset_loader=buffered was not honored for construction compression None",
                BaseVectorLoading.BUFFERED, BaseVectorLoading.forBenchmark(config("None", "PQ")));
    }
}
