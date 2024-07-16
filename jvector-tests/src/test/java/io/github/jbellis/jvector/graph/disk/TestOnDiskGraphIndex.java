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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.TestVectorGraph;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.github.jbellis.jvector.TestUtil.getNeighborNodes;
import static org.junit.Assert.*;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndex extends RandomizedTest {

    private Path testDirectory;

    private TestUtil.FullyConnectedGraphIndex fullyConnectedGraph;
    private TestUtil.RandomlyConnectedGraphIndex randomlyConnectedGraph;

    @Before
    public void setup() throws IOException {
        fullyConnectedGraph = new TestUtil.FullyConnectedGraphIndex(0, 6);
        randomlyConnectedGraph = new TestUtil.RandomlyConnectedGraphIndex(10, 4, getRandom());
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testSimpleGraphs() throws Exception {
        for (var graph : List.of(fullyConnectedGraph, randomlyConnectedGraph))
        {
            var outputPath = testDirectory.resolve("test_graph_" + graph.getClass().getSimpleName());
            var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size());
            TestUtil.writeGraph(graph, ravv, outputPath);
            try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
                 var onDiskGraph = OnDiskGraphIndex.load(marr::duplicate))
            {
                TestUtil.assertGraphEquals(graph, onDiskGraph);
                try (var onDiskView = onDiskGraph.getView()) {
                    validateVectors(onDiskView, ravv);
                }
            }
        }
    }

    @Test
    public void testRenumberingOnDelete() throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var original = TestUtil.buildSequentially(builder, ravv);

        // delete the first node
        builder.markNodeDeleted(0);
        builder.cleanup();
        builder.setEntryPoint(builder.getGraph().getIdUpperBound() - 1);

        // check
        assertEquals(2, original.size());
        var originalView = original.getView();
        // 1 -> 2
        assertEquals(1, getNeighborNodes(originalView, 1).size());
        assertTrue(getNeighborNodes(originalView, 1).contains(2));
        // 2 -> 1
        assertEquals(1, getNeighborNodes(originalView, 2).size());
        assertTrue(getNeighborNodes(originalView, 2).contains(1));

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = OnDiskGraphIndexWriter.sequentialRenumbering(original);
        assertEquals(2, oldToNewMap.size());
        assertEquals(0, (int) oldToNewMap.get(1));
        assertEquals(1, (int) oldToNewMap.get(2));

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = OnDiskGraphIndex.load(marr::duplicate);
             var onDiskView = onDiskGraph.getView())
        {
            // entry point renumbering
            assertNotNull(onDiskView.getVector(onDiskGraph.entryNode));
            // 0 -> 1
            assertTrue(getNeighborNodes(onDiskView, 0).contains(1));
            // 1 -> 0
            assertTrue(getNeighborNodes(onDiskView, 1).contains(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testReorderingRenumbering() throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var original = TestUtil.buildSequentially(builder, ravv);

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = new HashMap<>();
        oldToNewMap.put(0, 2);
        oldToNewMap.put(1, 1);
        oldToNewMap.put(2, 0);

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = OnDiskGraphIndex.load(marr::duplicate);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(onDiskView.getVector(0), ravv.getVector(2));
            assertEquals(onDiskView.getVector(1), ravv.getVector(1));
            assertEquals(onDiskView.getVector(2), ravv.getVector(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testReorderingWithHoles() throws IOException {
        // graph of 3 vectors
        var ravv = new TestVectorGraph.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var original = TestUtil.buildSequentially(builder, ravv);

        // create renumbering map
        Map<Integer, Integer> oldToNewMap = new HashMap<>();
        oldToNewMap.put(0, 2);
        oldToNewMap.put(1, 10);
        oldToNewMap.put(2, 0);

        // write the graph
        var outputPath = testDirectory.resolve("renumbered_graph");
        OnDiskGraphIndex.write(original, ravv, oldToNewMap, outputPath);
        // check that written graph ordinals match the new ones
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = OnDiskGraphIndex.load(marr::duplicate);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(onDiskView.getVector(0), ravv.getVector(2));
            assertEquals(onDiskView.getVector(10), ravv.getVector(1));
            assertEquals(onDiskView.getVector(2), ravv.getVector(0));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void validateVectors(OnDiskGraphIndex.View view, RandomAccessVectorValues ravv) {
        for (int i = 0; i < view.size(); i++) {
            assertEquals("Incorrect vector at " + i, view.getVector(i), ravv.getVector(i));
        }
    }

    @Test
    public void testLargeGraph() throws Exception
    {
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1_000_000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var ravv = new TestVectorGraph.CircularFloatVectorValues(graph.size());
        TestUtil.writeGraph(graph, ravv, outputPath);

        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = OnDiskGraphIndex.load(marr::duplicate);
             var cachedOnDiskGraph = new CachingGraphIndex(onDiskGraph))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            TestUtil.assertGraphEquals(graph, cachedOnDiskGraph);
            try (var onDiskView = onDiskGraph.getView();
                 var cachedOnDiskView = onDiskGraph.getView())
            {
                validateVectors(onDiskView, ravv);
                validateVectors(cachedOnDiskView, ravv);
            }
        }
    }

    @Test
    public void testV0Read() throws IOException {
        // using a random graph from testLargeGraph generated on old version
        var file = new File("resources/version0.odgi");
        try (var smr = new SimpleMappedReader(file.getAbsolutePath());
             var onDiskGraph = OnDiskGraphIndex.load(smr::duplicate);
             var onDiskView = onDiskGraph.getView())
        {
            assertEquals(32, onDiskGraph.maxDegree);
            assertEquals(2, onDiskGraph.version);
            assertEquals(100_000, onDiskGraph.size);
            assertEquals(2, onDiskGraph.dimension);
            assertEquals(99779, onDiskGraph.entryNode);
            assertEquals(EnumSet.of(FeatureId.INLINE_VECTORS), onDiskGraph.features.keySet());
            var actualNeighbors = getNeighborNodes(onDiskView, 12345);
            var expectedNeighbors = Set.of(67461, 9540, 85444, 13638, 89415, 21255, 73737, 46985, 71373, 47436, 94863, 91343, 27215, 59730, 69911, 91867, 89373, 6621, 59106, 98922, 69679, 47728, 60722, 56052, 28854, 38902, 21561, 20665, 41722, 57917, 34495, 5183);
            assertEquals(expectedNeighbors, actualNeighbors);
        }
    }

    @Test
    public void testV0Write() throws IOException {
        var fileIn = new File("resources/version0.odgi");
        var fileOut = File.createTempFile("version0", ".odgi");

        try (var smr = new SimpleMappedReader(fileIn.getAbsolutePath());
             var graph = OnDiskGraphIndex.load(smr::duplicate);
             var view = graph.getView())
        {
             try (var writer = new OnDiskGraphIndexWriter.Builder(graph, fileOut.toPath())
                     .withVersion(2)
                     .with(new InlineVectors(graph.dimension))
                     .build())
             {
                 writer.write(Feature.singleStateFactory(FeatureId.INLINE_VECTORS, nodeId -> new InlineVectors.State(view.getVector(nodeId))));
             }
        }

        // check that the contents match
        var contents1 = Files.readAllBytes(fileIn.toPath());
        var contents2 = Files.readAllBytes(fileOut.toPath());
        assertArrayEquals(contents1, contents2);
    }

    @Test
    public void testV0WriteIncremental() throws IOException {
        var fileIn = new File("resources/version0.odgi");
        var fileOut = File.createTempFile("version0", ".odgi");

        try (var smr = new SimpleMappedReader(fileIn.getAbsolutePath());
             var graph = OnDiskGraphIndex.load(smr::duplicate);
             var view = graph.getView())
        {
            try (var writer = new OnDiskGraphIndexWriter.Builder(graph, fileOut.toPath())
                    .withVersion(2)
                    .with(new InlineVectors(graph.dimension))
                    .build())
            {
                for (int i = 0; i < view.size(); i++) {
                    var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(view.getVector(i)));
                    writer.writeInline(i, state);
                }
                writer.write(Map.of());
            }
        }

        // check that the contents match
        var contents1 = Files.readAllBytes(fileIn.toPath());
        var contents2 = Files.readAllBytes(fileOut.toPath());
        assertArrayEquals(contents1, contents2);
    }

    @Test
    public void testIncrementalWrites() throws IOException {
        // generate 1000 node random graph
        var graph = new TestUtil.RandomlyConnectedGraphIndex(1000, 32, getRandom());
        var vectors = TestUtil.createRandomVectors(1000, 256);
        var ravv = new ListRandomAccessVectorValues(vectors, 256);

        // write out graph all at once
        var bulkPath = testDirectory.resolve("bulk_graph");
        OnDiskGraphIndex.write(graph, ravv, bulkPath);

        // write incrementally
        var incrementalPath = testDirectory.resolve("bulk_graph");
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, incrementalPath)
                .with(new InlineVectors(ravv.dimension()))
                .build())
        {
            // write inline vectors incrementally
            for (int i = 0; i < vectors.size(); i++) {
                var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(i)));
                writer.writeInline(i, state);
            }

            // write graph structure
            writer.write(Map.of());
        }

        // all-at-once and incremental builds should be identical on disk
        var bulkContents = Files.readAllBytes(bulkPath);
        var incrementalContents = Files.readAllBytes(incrementalPath);
        assertArrayEquals(bulkContents, incrementalContents);

        // write incrementally and add Fused ADC Feature
        var incrementalFadcPath = testDirectory.resolve("incremental_graph");
        var pq = ProductQuantization.compute(ravv, 64, 256, false);
        var pqv = (PQVectors) pq.createCompressedVectors(pq.encodeAll(ravv));
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, incrementalFadcPath)
                .with(new InlineVectors(ravv.dimension()))
                .with(new FusedADC(graph.maxDegree(), pq))
                .build())
        {
            // write inline vectors incrementally
            for (int i = 0; i < vectors.size(); i++) {
                var state = Feature.singleState(FeatureId.INLINE_VECTORS, new InlineVectors.State(ravv.getVector(i)));
                writer.writeInline(i, state);
            }
            // write graph structure, fused ADC
            writer.write(Feature.singleStateFactory(FeatureId.FUSED_ADC, i -> new FusedADC.State(graph.getView(), pqv, i)));
            writer.write(Map.of());
        }

        // graph and vectors should be identical
        try (var bulkMarr = new SimpleMappedReader(bulkPath.toAbsolutePath().toString());
             var bulkGraph = OnDiskGraphIndex.load(bulkMarr::duplicate);
             var incrementalMarr = new SimpleMappedReader(incrementalFadcPath.toAbsolutePath().toString());
             var incrementalGraph = OnDiskGraphIndex.load(incrementalMarr::duplicate);
             var incrementalView = incrementalGraph.getView())
        {
            assertTrue(OnDiskGraphIndex.areHeadersEqual(incrementalGraph, bulkGraph));
            TestUtil.assertGraphEquals(incrementalGraph, bulkGraph); // incremental and bulk graph should have same structure
            validateVectors(incrementalView, ravv); // inline vectors should be the same
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // write incrementally with LVQ and add Fused ADC feature
        var incrementalLvqPath = testDirectory.resolve("incremental_lvq_graph");
        var lvq = LocallyAdaptiveVectorQuantization.compute(ravv);
        var lvqFeature = new LVQ(lvq);

        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, incrementalLvqPath)
                .with(lvqFeature)
                .with(new FusedADC(graph.maxDegree(), pq))
                .build())
        {
            // write inline vectors incrementally
            for (int i = 0; i < vectors.size(); i++) {
                var state = Feature.singleState(FeatureId.LVQ, new LVQ.State(lvq.encode(ravv.getVector(i))));
                writer.writeInline(i, state);
            }

            // write graph structure, fused ADC
            writer.write(Feature.singleStateFactory(FeatureId.FUSED_ADC, i -> new FusedADC.State(graph.getView(), pqv, i)));
            writer.write(Map.of());
        }

        // graph and vectors should be identical
        try (var bulkMarr = new SimpleMappedReader(bulkPath.toAbsolutePath().toString());
             var bulkGraph = OnDiskGraphIndex.load(bulkMarr::duplicate);
             var incrementalMarr = new SimpleMappedReader(incrementalLvqPath.toAbsolutePath().toString());
             var incrementalGraph = OnDiskGraphIndex.load(incrementalMarr::duplicate))
        {
            assertTrue(OnDiskGraphIndex.areHeadersEqual(incrementalGraph, bulkGraph));
            TestUtil.assertGraphEquals(incrementalGraph, bulkGraph); // incremental and bulk graph should have same structure
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
