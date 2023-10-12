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

package io.github.jbellis.jvector.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphIndexTestCase;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static io.github.jbellis.jvector.TestUtil.getNeighborNodes;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestOnDiskGraphIndex extends RandomizedTest {

    private Path testDirectory;

    private TestUtil.FullyConnectedGraphIndex<float[]> fullyConnectedGraph;
    private TestUtil.RandomlyConnectedGraphIndex<float[]> randomlyConnectedGraph;

    @Before
    public void setup() throws IOException {
        fullyConnectedGraph = new TestUtil.FullyConnectedGraphIndex<>(0, 6);
        randomlyConnectedGraph = new TestUtil.RandomlyConnectedGraphIndex<>(10, 4, getRandom());
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
            var ravv = new GraphIndexTestCase.CircularFloatVectorValues(graph.size());
            TestUtil.writeGraph(graph, ravv, outputPath);
            try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
                 var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0))
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
        var ravv = new GraphIndexTestCase.CircularFloatVectorValues(3);
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var original = TestUtil.buildSequentially(builder, ravv);

        // delete the first node
        builder.markNodeDeleted(0);
        builder.cleanup();

        // check
        assertEquals(2, original.size());
        var originalView = original.getView();
        // 1 -> 2
        assertEquals(1, getNeighborNodes(originalView, 1).size());
        assertTrue(getNeighborNodes(originalView, 1).contains(2));
        // 2 -> 1
        assertEquals(1, getNeighborNodes(originalView, 2).size());
        assertTrue(getNeighborNodes(originalView, 2).contains(1));

        // check renumbered
        var renumbered = new RenumberingGraphIndex<>(original);
        var renumberedView = renumbered.getView();
        assertEquals(2, renumbered.size());
        // 0 -> 1
        assertEquals(1, getNeighborNodes(renumberedView, 0).size());
        assertTrue(getNeighborNodes(renumberedView, 0).contains(1));
        // 1 -> 0
        assertEquals(1, getNeighborNodes(renumberedView, 1).size());
        assertTrue(getNeighborNodes(renumberedView, 1).contains(0));

        // writing to disk should be the same as the renumbered
        var outputPath = testDirectory.resolve("large_graph");
        TestUtil.writeGraph(original, ravv, outputPath);
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0))
        {
            TestUtil.assertGraphEquals(renumbered, onDiskGraph);
        }
    }

    private static void validateVectors(GraphIndex.View<float[]> view, RandomAccessVectorValues<float[]> ravv) {
        for (int i = 0; i < view.size(); i++) {
            assertArrayEquals(view.getVector(i), ravv.vectorValue(i), 0.0f);
        }
    }

    @Test
    public void testLargeGraph() throws Exception
    {
        var graph = new TestUtil.RandomlyConnectedGraphIndex<float[]>(100_000, 16, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var ravv = new GraphIndexTestCase.CircularFloatVectorValues(graph.size());
        TestUtil.writeGraph(graph, ravv, outputPath);

        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0);
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
}
