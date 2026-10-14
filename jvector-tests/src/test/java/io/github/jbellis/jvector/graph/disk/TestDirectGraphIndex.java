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

package io.github.jbellis.jvector.graph.disk;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import org.junit.Test;

// PQ training uses the library's process-wide ForkJoin pools.
@com.carrotsearch.randomizedtesting.annotations.
ThreadLeakScope(com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope.Scope.NONE)
public class TestDirectGraphIndex extends LuceneTestCase {
    @Test
    public void testSearchAndResumeParity() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        try {
            Class.forName("io.github.jbellis.jvector.disk.LinuxDirectReader");
        } catch (ClassNotFoundException e) {
            assumeTrue("Native module is required", false);
        }
        var dir = Files.createTempDirectory("direct-search");
        try {
            // Cross-sector inline vectors plus fused PQ/NVQ records, including resident hierarchy.
            for (long cacheBytes : new long[] {0, 40000, 1024 * 1024})
                for (boolean fused : new boolean[] {false, true}) {
                    int dimension = fused ? 64 : 1536;
                    var vectors =
                            new ListRandomAccessVectorValues(TestUtil.createRandomVectors(256, dimension), dimension);
                    var graph = new TestUtil.RandomlyConnectedGraphIndex(
                            List.of(new CommonHeader.LayerInfo(64, 32), new CommonHeader.LayerInfo(16, 8),
                                    new CommonHeader.LayerInfo(4, 2)),
                            getRandom());
                    var path = dir.resolve("graph");
                    Files.deleteIfExists(path);
                    try (var view = graph.getView(); var builder = new OnDiskGraphIndexWriter.Builder(graph, path)
                                                                           .with(new InlineVectors(dimension))
                                                                           .build()) {
                        builder.write(Feature.singleStateFactory(
                                FeatureId.INLINE_VECTORS, n -> new InlineVectors.State(vectors.getVector(n))));
                    }
                    if (fused) {
                        Files.deleteIfExists(path);
                        var pq = ProductQuantization.compute(vectors, dimension / 8, 256, false);
                        var codes = (PQVectors) pq.encodeAll(vectors);
                        var nvq = NVQuantization.compute(vectors, 2);
                        try (var view = graph.getView(); var writer = new OnDiskGraphIndexWriter.Builder(graph, path)
                                                                              .with(new NVQ(nvq))
                                                                              .with(new FusedPQ(32, pq))
                                                                              .build()) {
                            Map<FeatureId, IntFunction<Feature.State>> states = Map.of(FeatureId.NVQ_VECTORS,
                                    n
                                    -> new NVQ.State(nvq.encode(vectors.getVector(n))),
                                    FeatureId.FUSED_PQ, n -> new FusedPQ.State(view, codes, n));
                            writer.write(states);
                        }
                    }
                    try (var mapped = new SimpleMappedReader.Supplier(path);
                            var direct = ReaderSupplierFactory.openDirect(path, 8, cacheBytes)) {
                        var a = OnDiskGraphIndex.load(mapped);
                        var b = OnDiskGraphIndex.load(direct);
                        try (var as = new GraphSearcher(a); var bs = new GraphSearcher(b)) {
                            var av = (OnDiskGraphIndex.View) as.getView();
                            var bv = (OnDiskGraphIndex.View) bs.getView();
                            for (int q = 200; q < 205; q++)
                                for (int k : new int[] {10, 20})
                                    for (var vsf : VectorSimilarityFunction.values()) {
                                        var query = vectors.getVector(q);
                                        var ap = fused ? new DefaultSearchScoreProvider(
                                                         av.approximateScoreFunctionFor(query, vsf),
                                                         av.rerankerFor(query, vsf))
                                                       : DefaultSearchScoreProvider.exact(query, vsf, av);
                                        var bp = fused ? new DefaultSearchScoreProvider(
                                                         bv.approximateScoreFunctionFor(query, vsf),
                                                         bv.rerankerFor(query, vsf))
                                                       : DefaultSearchScoreProvider.exact(query, vsf, bv);
                                        compare(as.search(ap, 10, k, 0f, 0f, Bits.ALL),
                                                bs.search(bp, 10, k, 0f, 0f, Bits.ALL));
                                        compare(as.resume(5, k), bs.resume(5, k));
                                        if (fused && q == 200 && k == 20) {
                                            Bits accepted = node -> node % 3 != 0;
                                            compare(as.search(ap, 10, k, 0f, Float.MAX_VALUE, accepted),
                                                    bs.search(bp, 10, k, 0f, Float.MAX_VALUE, accepted));
                                            compare(as.resume(5, k), bs.resume(5, k));
                                        }
                                    }
                        }
                    }
                }
        } finally {
            TestUtil.deleteQuietly(dir);
        }
    }
    private void compare(SearchResult a, SearchResult b) {
        assertEquals(a.getVisitedCount(), b.getVisitedCount());
        assertEquals(a.getExpandedCount(), b.getExpandedCount());
        assertEquals(a.getNodes().length, b.getNodes().length);
        for (int i = 0; i < a.getNodes().length; i++) {
            assertEquals(a.getNodes()[i].node, b.getNodes()[i].node);
            assertEquals(
                    Float.floatToRawIntBits(a.getNodes()[i].score), Float.floatToRawIntBits(b.getNodes()[i].score));
        }
    }

    @Test
    public void testExistingFormatsAndEveryStoredFeature() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        try {
            Class.forName("io.github.jbellis.jvector.disk.LinuxDirectReader");
        } catch (ClassNotFoundException e) {
            assumeTrue("Native module is required", false);
        }
        var dir = Files.createTempDirectory("direct-formats");
        try {
            int dimension = 24;
            var vectors = new ListRandomAccessVectorValues(TestUtil.createRandomVectors(256, dimension), dimension);
            var nvq = NVQuantization.compute(vectors, 2);
            var pq = ProductQuantization.compute(vectors, 3, 256, false);
            var codes = (PQVectors) pq.encodeAll(vectors);
            for (int version = 2; version <= OnDiskGraphIndex.CURRENT_VERSION; version++) {
                var layers = version < 4
                        ? List.of(new CommonHeader.LayerInfo(64, 16))
                        : List.of(new CommonHeader.LayerInfo(64, 16), new CommonHeader.LayerInfo(16, 8));
                var graph = new TestUtil.RandomlyConnectedGraphIndex(layers, getRandom());
                for (FeatureId id : GraphIndexFormatFactory.forVersion(version).getSupportedFeatures()) {
                    Feature feature;
                    switch (id) {
                        case INLINE_VECTORS:
                            feature = new InlineVectors(dimension);
                            break;
                        case NVQ_VECTORS:
                            feature = new NVQ(nvq);
                            break;
                        case SEPARATED_VECTORS:
                            feature = new SeparatedVectors(dimension, 0);
                            break;
                        case SEPARATED_NVQ:
                            feature = new SeparatedNVQ(nvq, 0);
                            break;
                        case FUSED_PQ:
                            feature = new FusedPQ(16, pq);
                            break;
                        default:
                            throw new AssertionError(id);
                    }
                    var path = dir.resolve("v" + version + "-" + id);
                    long start = 37; // Index embedded at a non-sector-aligned file offset.
                    var builder = new OnDiskGraphIndexWriter.Builder(graph, path).withStartOffset(start);
                    builder.withVersion(version).with(feature);
                    if (id == FeatureId.FUSED_PQ) builder.with(new InlineVectors(dimension));
                    try (var view = graph.getView(); var writer = builder.build()) {
                        var states = new java.util.EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
                        states.put(id, node -> {
                            switch (id) {
                                case INLINE_VECTORS:
                                case SEPARATED_VECTORS:
                                    return new InlineVectors.State(vectors.getVector(node));
                                case NVQ_VECTORS:
                                case SEPARATED_NVQ:
                                    return new NVQ.State(nvq.encode(vectors.getVector(node)));
                                case FUSED_PQ:
                                    return new FusedPQ.State(view, codes, node);
                                default:
                                    throw new AssertionError(id);
                            }
                        });
                        if (id == FeatureId.FUSED_PQ)
                            states.put(FeatureId.INLINE_VECTORS, node -> new InlineVectors.State(vectors.getVector(node)));
                        writer.write(states);
                    }
                    for (int depth : new int[] {0, 8}) {
                        try (var mapped = new SimpleMappedReader.Supplier(path);
                                var direct = ReaderSupplierFactory.openDirect(path, depth, 1024 * 1024)) {
                            var a = OnDiskGraphIndex.load(mapped, start, false);
                            var b = OnDiskGraphIndex.load(direct, start, false);
                            try (var av = a.getView(); var bv = b.getView()) {
                                for (int level = 0; level < layers.size(); level++) {
                                    for (int node = 0; node < layers.get(level).size; node++) {
                                        var an = av.getNeighborsIterator(level, node);
                                        var bn = bv.getNeighborsIterator(level, node);
                                        assertEquals(an.size(), bn.size());
                                        while (an.hasNext()) assertEquals(an.nextInt(), bn.nextInt());
                                        assertFalse(bn.hasNext());
                                    }
                                }
                                for (int node = 0; node < 64; node++) {
                                    byte[] expected = new byte[feature.featureSize()];
                                    byte[] actual = new byte[expected.length];
                                    av.featureReaderForNode(node, id).readFully(expected);
                                    bv.featureReaderForNode(node, id).readFully(actual);
                                    assertArrayEquals(expected, actual);
                                }
                            }
                        }
                    }
                }
            }
        } finally {
            TestUtil.deleteQuietly(dir);
        }
    }
}
