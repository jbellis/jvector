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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.*;

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

    private static <T> void validateGraph(GraphIndex.View<T> expectedView, GraphIndex.View<T> actualView) {
        assertEquals(expectedView.size(), actualView.size());
        assertEquals(expectedView.entryNode(), actualView.entryNode());

        var nodes = expectedView.getSortedNodes();
        assertArrayEquals(nodes, actualView.getSortedNodes());

        // For each node, check its neighbors
        for (int j : nodes) {
            var expectedNeighbors = expectedView.getNeighborsIterator(j);
            var actualNeighbors = actualView.getNeighborsIterator(j);
            assertEquals(expectedNeighbors.size(), actualNeighbors.size());
            while (expectedNeighbors.hasNext()) {
                assertEquals(expectedNeighbors.nextInt(), actualNeighbors.nextInt());
            }
            assertFalse(actualNeighbors.hasNext());
        }
    }

    @Test
    public void testSimpleGraphs() throws Exception {
        for (var graph : List.of(fullyConnectedGraph, randomlyConnectedGraph))
        {
            var outputPath = testDirectory.resolve("test_graph_" + graph.getClass().getSimpleName());
            var ravv = new GraphIndexTestCase.CircularFloatVectorValues(graph.size());
            TestUtil.writeGraph(graph, ravv, outputPath);
            try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
                 var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0);
                 var onDiskView = onDiskGraph.getView())
            {
                validateGraph(graph.getView(), onDiskView);
                validateVectors(onDiskView, ravv);
            }
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
             var onDiskView = onDiskGraph.getView();
             var cachedOnDiskGraph = new CachingGraphIndex(onDiskGraph);
             var cachedOnDiskView = cachedOnDiskGraph.getView())
        {
            validateGraph(graph.getView(), onDiskView);
            validateGraph(graph.getView(), cachedOnDiskView);
            validateVectors(onDiskView, ravv);
            validateVectors(cachedOnDiskView, ravv);
        }
    }
}
