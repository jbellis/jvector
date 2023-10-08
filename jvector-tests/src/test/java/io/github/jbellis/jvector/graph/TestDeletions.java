package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import static io.github.jbellis.jvector.graph.GraphIndexTestCase.createRandomFloatVectors;
import static org.junit.Assert.assertNotEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeletions extends LuceneTestCase {
    @Test
    public void testSimple() {
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var graph = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f).build();
        int n = getRandom().nextInt(ravv.size());
        graph.markDeleted(n);
        for (int i = 0; i < 100; i++) {
            var v = TestUtil.randomVector(getRandom(), dimension);
            assertNotFound(n, 3, v, ravv, graph);
        }
        var v = ravv.vectorValue(n);
        assertNotFound(n, ravv.size(), v, ravv, graph);
    }

    private static void assertNotFound(int n, int topK, float[] v, MockVectorValues ravv, OnHeapGraphIndex<float[]> graph) {
        var results = GraphSearcher.search(v, topK, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        for (var ns : results.getNodes()) {
            assertNotEquals(n, ns.node);
        }
    }
}
