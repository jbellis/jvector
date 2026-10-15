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

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.graph.disk.feature.InlineVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.quantization.PQVectors;
import io.github.jbellis.jvector.quantization.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Exercises writing and reading graphs whose on-disk ordinal space is non-contiguous ("holes"),
 * as produced by Cassandra SAI's {@code ONE_TO_MANY} postings structure.
 * See <a href="https://github.com/datastax/jvector/issues/746">jvector#746</a>.
 * <p>
 * In {@code ONE_TO_MANY}, each distinct vector is a graph node whose on-disk ordinal equals the
 * segment row-id of the first row that has that vector. Rows that duplicate an earlier vector
 * ("extra" rows) are not graph nodes; their row-ids become {@link OrdinalMapper#OMITTED} holes in
 * the on-disk ordinal space. Cassandra writes these graphs with {@link OnDiskGraphIndexWriter}
 * (memtable flush and compaction) or {@link OnDiskParallelGraphIndexWriter} (compaction), using
 * jvector format version 2 (SAI DC, EB), 4 (SAI EC, ED) or 6 (SAI FA and later).
 * <p>
 * Each test checks a single invariant so that the full set of failures across versions and
 * writers is visible in one run. Several are expected to fail until #746 is fixed:
 * <ul>
 *   <li>v2/v3 do not persist {@code idUpperBound}; it is read back as {@code size(0)}, so the
 *       header is wrong and {@link OnDiskGraphIndex#getNodes(int)} scans too few L0 records.</li>
 *   <li>{@link OnDiskGraphIndex} does not override {@link ImmutableGraphIndex#containsNode(int)},
 *       whose default is {@code nodeId < size()}, and {@code View.contains(0, n)} is
 *       {@code n < idUpperBound}; neither recognizes holes, in any version.</li>
 * </ul>
 */
@RunWith(Parameterized.class)
public class TestSparseOrdinalMapping {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction VSF = VectorSimilarityFunction.EUCLIDEAN;

    enum WriterKind { SEQUENTIAL, PARALLEL }

    @Parameterized.Parameters(name = "v{0} {1} trailingHole={2}")
    public static Collection<Object[]> parameters() {
        var params = new ArrayList<Object[]>();
        for (int version = 2; version <= OnDiskGraphIndex.CURRENT_VERSION; version++) {
            for (var writer : WriterKind.values()) {
                for (boolean trailingHole : new boolean[] {false, true}) {
                    params.add(new Object[] {version, writer, trailingHole});
                }
            }
        }
        return params;
    }

    private final int version;
    private final WriterKind writerKind;
    // Cassandra's BiMapMapper / OmissionAwareIdentityMapper set maxOrdinal to the largest
    // representative row-id, so the last slot is never a hole. The example in #746 uses
    // maxOrdinal = maxRowId, which leaves a hole in the final L0 record; cover both.
    private final boolean trailingHole;

    private Path testDirectory;

    public TestSparseOrdinalMapping(int version, WriterKind writerKind, boolean trailingHole) {
        this.version = version;
        this.writerKind = writerKind;
        this.trailingHole = trailingHole;
    }

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    // ---------------------------------------------------------------------------------------------
    // The 6-row example from #746:
    //
    //   row-id | vector | role                  | on-disk ordinal
    //   -------+--------+-----------------------+-----------------
    //     0    |   A    | representative        | 0
    //     1    |   B    | representative        | 1
    //     2    |   A    | extra (dup of row 0)  | hole
    //     3    |   C    | representative        | 3
    //     4    |   D    | representative        | 4
    //     5    |   B    | extra (dup of row 1)  | hole
    //
    // The graph is built over the 4 distinct vectors with dense ordinals 0..3 (A, B, C, D).
    // ---------------------------------------------------------------------------------------------

    private static final int[] EXAMPLE_OLD_TO_NEW = {0, 1, 3, 4};
    private static final Set<Integer> EXAMPLE_NODES = new TreeSet<>(List.of(0, 1, 3, 4));
    private static final Set<Integer> EXAMPLE_HOLES = Set.of(2, 5);

    private int exampleMaxOrdinal() {
        return trailingHole ? 5 : 4;
    }

    @Test
    public void headerRecordsIdUpperBoundAndLayerSize() throws IOException {
        try (var fixture = writeExample()) {
            assertEquals("size(0) should count only real nodes", 4, fixture.graph.size(0));
            assertEquals("idUpperBound should be maxOrdinal + 1, including holes",
                         exampleMaxOrdinal() + 1, fixture.graph.getIdUpperBound());
            assertEquals("View.getIdUpperBound should match the graph",
                         exampleMaxOrdinal() + 1, fixture.view.getIdUpperBound());
        }
    }

    @Test
    public void getNodesReturnsEachRealNodeExactlyOnce() throws IOException {
        try (var fixture = writeExample()) {
            var nodes = collect(fixture.graph.getNodes(0));
            nodes.sort(Integer::compare);
            assertEquals(List.of(0, 1, 3, 4), nodes);
        }
    }

    @Test
    public void vectorsAreReadableAtMappedOrdinals() throws IOException {
        try (var fixture = writeExample()) {
            for (int old = 0; old < EXAMPLE_OLD_TO_NEW.length; old++) {
                int ordinal = EXAMPLE_OLD_TO_NEW[old];
                assertEquals("vector at on-disk ordinal " + ordinal,
                             fixture.uniqueVectors.getVector(old), fixture.view.getVector(ordinal));
            }
        }
    }

    @Test
    public void edgesAndEntryNodeNeverReferenceHoles() throws IOException {
        try (var fixture = writeExample()) {
            assertTrue("entry node " + fixture.view.entryNode(),
                       EXAMPLE_NODES.contains(fixture.view.entryNode().node));
            for (int level = 0; level <= fixture.graph.getMaxLevel(); level++) {
                for (var it = fixture.graph.getNodes(level); it.hasNext(); ) {
                    int node = it.nextInt();
                    for (int neighbor : collect(fixture.view.getNeighborsIterator(level, node))) {
                        assertTrue(String.format("level %d: node %d -> %d", level, node, neighbor),
                                   EXAMPLE_NODES.contains(neighbor));
                    }
                }
            }
        }
    }

    @Test
    public void containsNodeExcludesHoles() throws IOException {
        try (var fixture = writeExample()) {
            for (int n : EXAMPLE_NODES) {
                assertTrue("containsNode(" + n + ")", fixture.graph.containsNode(n));
                assertTrue("view.contains(0, " + n + ")", fixture.view.contains(0, n));
            }
            for (int n : EXAMPLE_HOLES) {
                assertFalse("containsNode(" + n + ") on a hole", fixture.graph.containsNode(n));
                assertFalse("view.contains(0, " + n + ") on a hole", fixture.view.contains(0, n));
            }
        }
    }

    @Test
    public void searchReturnsOnlyRealNodes() throws IOException {
        try (var fixture = writeExample()) {
            for (int old = 0; old < EXAMPLE_OLD_TO_NEW.length; old++) {
                var query = fixture.uniqueVectors.getVector(old);
                var result = GraphSearcher.search(query, 4, fixture.view, VSF, fixture.graph, Bits.ALL);
                var found = Arrays.stream(result.getNodes()).map(ns -> ns.node).collect(Collectors.toList());
                assertEquals("top hit for vector " + old, EXAMPLE_OLD_TO_NEW[old], (int) found.get(0));
                assertEquals("all nodes, no holes, no duplicates", EXAMPLE_NODES, new TreeSet<>(found));
                assertEquals(4, found.size());
            }
        }
    }

    /**
     * {@link OnDiskGraphIndexWriter#sequentialRenumbering} (the default mapper used when re-writing
     * a graph) relies on {@code getIdUpperBound()} and {@code containsNode()} of the source graph.
     * Re-writing a sparse on-disk graph should keep exactly its real nodes.
     */
    @Test
    public void sequentialRenumberingOfLoadedGraphKeepsRealNodes() throws IOException {
        try (var fixture = writeExample()) {
            var renumbering = OnDiskGraphIndexWriter.sequentialRenumbering(fixture.graph);
            assertEquals(EXAMPLE_NODES, new TreeSet<>(renumbering.keySet()));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // A larger segment shaped like a Cassandra ONE_TO_MANY compaction: ~20% of rows duplicate an
    // earlier vector. Hierarchical where the format allows it, and with FusedPQ on v6 (the
    // configuration SAI V8 writes).
    // ---------------------------------------------------------------------------------------------

    @Test
    public void cassandraShapedSegmentRoundTripsAndSearches() throws IOException {
        var random = new Random(746);
        int dimension = 32;
        int rows = 1500;
        var segment = Segment.generate(random, rows, dimension, 0.2);
        boolean hierarchy = version >= 4;
        boolean fused = version >= 6;

        var builder = new GraphIndexBuilder(segment.uniqueVectors, VSF, 16, 100, 1.2f, 1.2f, hierarchy);
        var heapGraph = TestUtil.buildSequentially(builder, segment.uniqueVectors);
        var mapper = new SparseMapper(segment.oldToNew, segment.maxOrdinal);

        ProductQuantization pq = null;
        PQVectors pqv = null;
        if (fused) {
            pq = ProductQuantization.compute(segment.uniqueVectors, 8, 256, false);
            pqv = (PQVectors) pq.encodeAll(segment.uniqueVectors);
        }

        var path = testDirectory.resolve("segment");
        try (var heapView = heapGraph.getView()) {
            var suppliers = new EnumMap<FeatureId, IntFunction<Feature.State>>(FeatureId.class);
            suppliers.put(FeatureId.INLINE_VECTORS, old -> new InlineVectors.State(segment.uniqueVectors.getVector(old)));
            if (fused) {
                var codes = pqv;
                suppliers.put(FeatureId.FUSED_PQ, old -> new FusedPQ.State(heapView, codes, old));
            }
            writeGraph(heapGraph, mapper, dimension, fused ? new FusedPQ(heapGraph.maxDegree(), pq) : null, path, suppliers);
        }

        try (var readerSupplier = new SimpleMappedReader.Supplier(path);
             var graph = OnDiskGraphIndex.load(readerSupplier);
             var view = graph.getView())
        {
            Set<Integer> expectedNodes = Arrays.stream(segment.oldToNew).boxed().collect(Collectors.toSet());

            assertEquals(segment.uniqueVectors.size(), graph.size(0));

            for (int old = 0; old < segment.oldToNew.length; old++) {
                assertEquals(segment.uniqueVectors.getVector(old), view.getVector(segment.oldToNew[old]));
            }

            // search with a perturbed copy of a sample of the stored vectors and compare to brute force
            var searcher = new GraphSearcher(graph);
            int topK = 10;
            int queries = 100;
            int hits = 0;
            for (int q = 0; q < queries; q++) {
                var query = segment.uniqueVectors.getVector(random.nextInt(segment.uniqueVectors.size())).copy();
                for (int d = 0; d < dimension; d++) {
                    query.set(d, query.get(d) + 0.01f * (float) random.nextGaussian());
                }

                SearchResult result;
                if (fused) {
                    // the fused score function must come from the searcher's own view: edge loading
                    // reads the neighbor list that getPackedNeighbors() loaded into that same view
                    var searchView = (ImmutableGraphIndex.ScoringView) searcher.getView();
                    var ssp = new DefaultSearchScoreProvider(searchView.approximateScoreFunctionFor(query, VSF),
                                                             searchView.rerankerFor(query, VSF));
                    result = searcher.search(ssp, topK, 4 * topK, 0f, 0f, Bits.ALL);
                } else {
                    result = GraphSearcher.search(query, topK, 4 * topK, view, VSF, graph, Bits.ALL);
                }

                var found = Arrays.stream(result.getNodes()).map(ns -> ns.node).collect(Collectors.toList());
                for (int node : found) {
                    assertTrue("search returned non-node ordinal " + node, expectedNodes.contains(node));
                }
                assertEquals("duplicate results " + found, found.size(), new HashSet<>(found).size());
                hits += overlap(found, segment.bruteForce(query, topK));
            }
            double recall = hits / (double) (queries * topK);
            assertTrue("recall " + recall, recall >= 0.9);

            // checked last so that a header problem doesn't mask the search results above
            assertEquals("idUpperBound", segment.maxOrdinal + 1, graph.getIdUpperBound());
            var nodes = collect(graph.getNodes(0));
            assertEquals("getNodes(0) contents", expectedNodes, new HashSet<>(nodes));
            assertEquals("getNodes(0) count", expectedNodes.size(), nodes.size());
        }
    }

    // ---------------------------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------------------------

    private ExampleFixture writeExample() throws IOException {
        // Points on a line, so that the built graph is a chain A-B-C-D and every node is reachable.
        // (Four mutually equidistant vectors can leave a node with no incoming edges.)
        var a = vts.createFloatVector(new float[] {0, 0});
        var b = vts.createFloatVector(new float[] {1, 0});
        var c = vts.createFloatVector(new float[] {2.5f, 0});
        var d = vts.createFloatVector(new float[] {4, 0});
        var uniqueVectors = new ListRandomAccessVectorValues(List.of(a, b, c, d), 2);

        // hierarchy requires v4+
        var builder = new GraphIndexBuilder(uniqueVectors, VSF, 2, 10, 1.2f, 1.2f, version >= 4);
        var heapGraph = TestUtil.buildSequentially(builder, uniqueVectors);
        // fixture sanity check: on the heap (no remapping) every node is found
        for (int old = 0; old < uniqueVectors.size(); old++) {
            var result = GraphSearcher.search(uniqueVectors.getVector(old), 4, uniqueVectors, VSF, heapGraph, Bits.ALL);
            assertEquals("heap graph search should find all nodes", 4, result.getNodes().length);
        }
        var mapper = new SparseMapper(EXAMPLE_OLD_TO_NEW, exampleMaxOrdinal());

        var path = testDirectory.resolve("example");
        var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                                                   old -> new InlineVectors.State(uniqueVectors.getVector(old)));
        writeGraph(heapGraph, mapper, uniqueVectors.dimension(), null, path, suppliers);

        var readerSupplier = new SimpleMappedReader.Supplier(path);
        var graph = OnDiskGraphIndex.load(readerSupplier);
        return new ExampleFixture(readerSupplier, graph, uniqueVectors);
    }

    private void writeGraph(ImmutableGraphIndex graph,
                            OrdinalMapper mapper,
                            int dimension,
                            Feature extraFeature,
                            Path path,
                            Map<FeatureId, IntFunction<Feature.State>> suppliers) throws IOException
    {
        AbstractGraphIndexWriter.Builder<? extends RandomAccessOnDiskGraphIndexWriter, ?> builder =
                writerKind == WriterKind.SEQUENTIAL
                ? new OnDiskGraphIndexWriter.Builder(graph, path)
                : new OnDiskParallelGraphIndexWriter.Builder(graph, path);
        builder.withVersion(version)
               .withMapper(mapper)
               .with(new InlineVectors(dimension));
        if (extraFeature != null) {
            builder.with(extraFeature);
        }
        try (var writer = builder.build()) {
            writer.write(suppliers);
        }
    }

    private static List<Integer> collect(NodesIterator it) {
        var result = new ArrayList<Integer>();
        while (it.hasNext()) {
            result.add(it.nextInt());
        }
        return result;
    }

    private static int overlap(List<Integer> a, List<Integer> b) {
        var set = new HashSet<>(b);
        return (int) a.stream().filter(set::contains).count();
    }

    private static final class ExampleFixture implements AutoCloseable {
        final SimpleMappedReader.Supplier readerSupplier;
        final OnDiskGraphIndex graph;
        final OnDiskGraphIndex.View view;
        final RandomAccessVectorValues uniqueVectors;

        ExampleFixture(SimpleMappedReader.Supplier readerSupplier, OnDiskGraphIndex graph, RandomAccessVectorValues uniqueVectors) {
            this.readerSupplier = readerSupplier;
            this.graph = graph;
            this.view = graph.getView();
            this.uniqueVectors = uniqueVectors;
        }

        @Override
        public void close() throws IOException {
            view.close();
            graph.close();
            readerSupplier.close();
        }
    }

    /**
     * Rows of a segment, where some rows repeat the vector of an earlier row. Graph (old) ordinals
     * are assigned densely in order of first appearance; the on-disk (new) ordinal of each unique
     * vector is the row-id of its first appearance, as in Cassandra's {@code ONE_TO_MANY}.
     */
    private static final class Segment {
        final ListRandomAccessVectorValues uniqueVectors;
        final int[] oldToNew;
        final int maxOrdinal;

        private Segment(ListRandomAccessVectorValues uniqueVectors, int[] oldToNew) {
            this.uniqueVectors = uniqueVectors;
            this.oldToNew = oldToNew;
            this.maxOrdinal = Arrays.stream(oldToNew).max().orElse(-1);
        }

        static Segment generate(Random random, int rows, int dimension, double duplicateRate) {
            var vectors = new ArrayList<VectorFloat<?>>();
            var firstRowIds = new ArrayList<Integer>();
            for (int row = 0; row < rows; row++) {
                if (row > 0 && random.nextDouble() < duplicateRate) {
                    continue; // extra row: reuses an existing vector, leaves a hole at this row-id
                }
                vectors.add(TestUtil.randomVector(random, dimension));
                firstRowIds.add(row);
            }
            return new Segment(new ListRandomAccessVectorValues(vectors, dimension),
                               firstRowIds.stream().mapToInt(Integer::intValue).toArray());
        }

        /** exact top-k, in on-disk ordinals */
        List<Integer> bruteForce(VectorFloat<?> query, int topK) {
            return IntStream.range(0, uniqueVectors.size())
                            .boxed()
                            .sorted((i, j) -> Float.compare(VSF.compare(query, uniqueVectors.getVector(j)),
                                                            VSF.compare(query, uniqueVectors.getVector(i))))
                            .limit(topK)
                            .map(i -> oldToNew[i])
                            .collect(Collectors.toList());
        }
    }

    /**
     * Mirrors Cassandra's {@code V5VectorPostingsWriter.BiMapMapper}: a bijection between dense
     * graph ordinals and sparse on-disk ordinals, with an explicit {@code maxOrdinal} and
     * {@link #OMITTED} for every on-disk ordinal that is not a node.
     */
    private static final class SparseMapper implements OrdinalMapper {
        private final int[] oldToNew;
        private final Map<Integer, Integer> newToOld = new HashMap<>();
        private final int maxOrdinal;

        SparseMapper(int[] oldToNew, int maxOrdinal) {
            this.oldToNew = oldToNew;
            this.maxOrdinal = maxOrdinal;
            for (int old = 0; old < oldToNew.length; old++) {
                newToOld.put(oldToNew[old], old);
            }
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew[oldOrdinal];
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld.getOrDefault(newOrdinal, OMITTED);
        }
    }
}
