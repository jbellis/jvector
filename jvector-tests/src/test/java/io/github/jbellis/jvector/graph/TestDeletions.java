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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;

import static io.github.jbellis.jvector.TestUtil.assertGraphEquals;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeletions extends LuceneTestCase {
    @Test
    public void testMarkDeleted() {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete a random entry
        int n = getRandom().nextInt(ravv.size());
        builder.markNodeDeleted(n);
        // check that searching for random vectors never results in the deleted one
        for (int i = 0; i < 100; i++) {
            var v = TestUtil.randomVector(getRandom(), dimension);
            var results = GraphSearcher.search(v, 3, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            for (var ns : results.getNodes()) {
                assertNotEquals(n, ns.node);
            }
        }
        // check that asking for the entire graph back still doesn't surface the deleted one
        var v = ravv.getVector(n);
        var results = GraphSearcher.search(v, ravv.size(), ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(GraphIndex.prettyPrint(graph), ravv.size() - 1, results.getNodes().length);
        for (var ns : results.getNodes()) {
            assertNotEquals(n, ns.node);
        }
    }

    @Test
    public void testCleanup() throws IOException {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete all nodes that connect to a random node
        int nodeToIsolate = getRandom().nextInt(ravv.size());
        int nDeleted = 0;
        try (var view = graph.getView()) {
            for (var i = 0; i < view.size(); i++) {
                for (var it = view.getNeighborsIterator(i); it.hasNext(); ) {
                    if (nodeToIsolate == it.nextInt()) {
                        builder.markNodeDeleted(i);
                        nDeleted++;
                        break;
                    }
                }
            }
        }
        assertNotEquals(0, nDeleted);

        // cleanup removes the deleted nodes
        builder.cleanup();
        assertEquals(ravv.size() - nDeleted, graph.size());

        // cleanup should have added new connections to the node that would otherwise have been disconnected
        var v = ravv.getVector(nodeToIsolate).copy();
        var results = GraphSearcher.search(v, 10, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(nodeToIsolate, results.getNodes()[0].node);

        // check that we can save and load the graph with "holes" from the deletion
        var testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        var outputPath = testDirectory.resolve("on_heap_graph");
        try (var out = TestUtil.openDataOutputStream(outputPath)) {
            graph.save(out);
        }
        var b2 = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString())) {
            b2.load(marr);
        }
        var reloadedGraph = b2.getGraph();
        assertGraphEquals(graph, reloadedGraph);
    }

    @Test
    public void testMarkingAllNodesAsDeleted() {
        // build graph
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // mark all deleted
        for (var i = 0; i < graph.size(); i++) {
            graph.markDeleted(i);
        }

        // removeDeletedNodes should leave the graph empty
        builder.removeDeletedNodes();
        assertEquals(0, graph.size());
        assertEquals(OnHeapGraphIndex.NO_ENTRY_POINT, graph.entry());
    }

    @Test
    public void testNoPathToLiveNodesWhenRemovingDeletedNodes() throws IOException {
        // build a graph that has no path to nodes that won't be deleted
        // from the entry point.
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(5, dimension, getRandom()));
        try(var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f)) {
            var graph = builder.getGraph();

            var na0 = new NodeArray(dimension);
            na0.addInOrder(1, 0.5f);
            na0.addInOrder(2, 0.5f);
            graph.addNode(0, na0);

            var na1 = new NodeArray(dimension);
            na1.addInOrder(0, 0.5f);
            na1.addInOrder(2, 0.5f);
            graph.addNode(1, na1);

            var na2 = new NodeArray(dimension);
            na2.addInOrder(0, 0.5f);
            na2.addInOrder(1, 0.5f);
            graph.addNode(2, na2);

            var na3 = new NodeArray(dimension);
            na3.addInOrder(0, 0.5f);
            na3.addInOrder(2, 0.5f);
            graph.addNode(3, na3);

            var na4 = new NodeArray(dimension);
            na4.addInOrder(0, 0.5f);
            na4.addInOrder(2, 0.5f);
            graph.addNode(4, na4);

            graph.updateEntryNode(1);

            builder.markNodeDeleted(0);
            builder.markNodeDeleted(1);
            builder.markNodeDeleted(2);

            // node 3 and 4 are live, but there are no edges pointing to them
            builder.removeDeletedNodes();

            assertEquals(2, graph.size());
            assertNotEquals(OnHeapGraphIndex.NO_ENTRY_POINT, graph.entry());
        }
    }
}
