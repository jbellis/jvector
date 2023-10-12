package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.util.Arrays;

import static io.github.jbellis.jvector.graph.GraphIndexTestCase.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeletions extends LuceneTestCase {
    @Test
    public void testMarkDeleted() {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete a random entry
        int n = getRandom().nextInt(ravv.size());
        builder.markNodeDeleted(n);
        // check that searching for random vectors never results in the deleted one
        for (int i = 0; i < 100; i++) {
            var v = TestUtil.randomVector(getRandom(), dimension);
            var results = GraphSearcher.search(v, 3, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            for (var ns : results.getNodes()) {
                assertNotEquals(n, ns.node);
            }
        }
        // check that asking for the entire graph back still doesn't surface the deleted one
        var v = ravv.vectorValue(n);
        var results = GraphSearcher.search(v, ravv.size(), ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(GraphIndex.prettyPrint(graph), ravv.size() - 1, results.getNodes().length);
        for (var ns : results.getNodes()) {
            assertNotEquals(n, ns.node);
        }
    }

    @Test
    public void testCleanup() {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete all nodes that connect to a random node
        int nodeToIsolate = getRandom().nextInt(ravv.size());
        var view = graph.getView();
        int nDeleted = 0;
        for (var i = 0; i < view.size(); i++) {
            for (var it = graph.getView().getNeighborsIterator(i); it.hasNext(); ) {
                if (nodeToIsolate == it.nextInt()) {
                    builder.markNodeDeleted(i);
                    nDeleted++;
                    break;
                }
            }
        }
        assertNotEquals(0, nDeleted);

        // cleanup removes the deleted nodes
        builder.cleanup();
        assertEquals(ravv.size() - nDeleted, graph.size());

        // cleanup should have added new connections to the node that would otherwise have been disconnected
        var v = Arrays.copyOf(ravv.vectorValue(nodeToIsolate), ravv.dimension);
        var results = GraphSearcher.search(v, 10, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(nodeToIsolate, results.getNodes()[0].node);
    }
}
