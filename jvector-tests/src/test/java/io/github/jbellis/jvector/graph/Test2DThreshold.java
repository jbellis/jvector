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
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class Test2DThreshold extends LuceneTestCase {
    @Test
    public void testThreshold() throws IOException {
        var R = getRandom();
        // generate 2D vectors
        float[][] vectors = new float[10000][2];
        for (int i = 0; i < vectors.length; i++) {
            vectors[i][0] = R.nextFloat();
            vectors[i][1] = R.nextFloat();
        }

        var ravv = new ListRandomAccessVectorValues(List.of(vectors), 2);
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN, 6, 32, 1.2f, 1.4f);
        var onHeapGraph = builder.build();

        // test raw vectors
        var searcher = new GraphSearcher.Builder<>(onHeapGraph.getView()).build();
        for (int i = 0; i < 10; i++) {
            TestParams tp = createTestParams(vectors);

            NodeSimilarity.ExactScoreFunction sf = j -> VectorSimilarityFunction.EUCLIDEAN.compare(tp.q, ravv.vectorValue(j));
            var result = searcher.search(sf, null, vectors.length, tp.th, Bits.ALL);

            assert result.getVisitedCount() < vectors.length : "visited all vectors for threshold " + tp.th;
            assert result.getNodes().length >= 0.9 * tp.exactCount : "returned " + result.getNodes().length + " nodes for threshold " + tp.th + " but should have returned at least " + tp.exactCount;
        }

        // test compressed
        Path outputPath = Files.createTempFile("graph", ".jvector");
        TestUtil.writeGraph(onHeapGraph, ravv, outputPath);
        var pq = ProductQuantization.compute(ravv, ravv.dimension(), false);
        var cv = new PQVectors(pq, pq.encodeAll(List.of(vectors)));

        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0))
        {
            for (int i = 0; i < 10; i++) {
                TestParams tp = createTestParams(vectors);
                searcher = new GraphSearcher.Builder<>(onDiskGraph.getView()).build();
                NodeSimilarity.ReRanker reranker = (j) -> VectorSimilarityFunction.EUCLIDEAN.compare(tp.q, ravv.vectorValue(j));
                var asf = cv.approximateScoreFunctionFor(tp.q, VectorSimilarityFunction.EUCLIDEAN);
                var result = searcher.search(asf, reranker, vectors.length, tp.th, Bits.ALL);

                assert result.getVisitedCount() < vectors.length : "visited all vectors for threshold " + tp.th;
            }
        }

    }

    // it's not an interesting test if all the vectors are within the threshold
    private TestParams createTestParams(float[][] vectors) {
        var R = getRandom();

        long exactCount;
        float[] q;
        float th;
        do {
            q = new float[]{R.nextFloat(), R.nextFloat()};
            th = (float) (0.2 + 0.8 * R.nextDouble());
            float[] finalQ = q;
            float finalTh = th;
            exactCount = Arrays.stream(vectors).filter(v -> VectorSimilarityFunction.EUCLIDEAN.compare(finalQ, v) >= finalTh).count();
        } while (!(exactCount < vectors.length * 0.8));

        return new TestParams(exactCount, q, th);
    }

    private static class TestParams {
        public final long exactCount;
        public final float[] q;
        public final float th;

        public TestParams(long exactCount, float[] q, float th) {
            this.exactCount = exactCount;
            this.q = q;
            this.th = th;
        }
    }
}
