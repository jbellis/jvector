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

package io.github.jbellis.jvector.pq;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.GraphIndexTestCase;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.abs;
import static java.lang.Math.log;
import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestCompressedVectors extends RandomizedTest {
    @Test
    public void testSaveLoad() throws Exception {
        // Generate a PQ for random 2D vectors
        var vectors = createRandomVectors(512, 2);
        var pq = ProductQuantization.compute(new ListRandomAccessVectorValues(vectors, 2), 1, false);

        // Compress the vectors
        var compressed = pq.encodeAll(vectors);
        var cv = new CompressedVectors(pq, compressed);

        // Write compressed vectors
        File cvFile = File.createTempFile("pqtest", ".cv");
        try (var out = new DataOutputStream(new FileOutputStream(cvFile))) {
            cv.write(out);
        }
        // Read compressed vectors
        try (var in = new SimpleMappedReader(cvFile.getAbsolutePath())) {
            var cv2 = CompressedVectors.load(in, 0);
            assertEquals(cv, cv2);
        }
    }

    private static List<float[]> createRandomVectors(int count, int dimension) {
        return IntStream.range(0, count).mapToObj(i -> TestUtil.randomVector(getRandom(), dimension)).collect(Collectors.toList());
    }

    private void testEncodings(int dimension, int codebooks) {
        // Generate a PQ for random 2D vectors
        var vectors = createRandomVectors(512, dimension);
        var pq = ProductQuantization.compute(new ListRandomAccessVectorValues(vectors, dimension), codebooks, false);

        // Compress the vectors
        var compressed = pq.encodeAll(vectors);
        var cv = new CompressedVectors(pq, compressed);

        // compare the encoded similarities to the raw
        for (var vsf : List.of(VectorSimilarityFunction.EUCLIDEAN, VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.COSINE)) {
            double delta = 0;
            for (int i = 0; i < 10; i++) {
                var q = TestUtil.randomVector(getRandom(), dimension);
                var f = cv.approximateScoreFunctionFor(q, vsf);
                for (int j = 0; j < vectors.size(); j++) {
                    delta += abs(f.similarityTo(j) - vsf.compare(q, vectors.get(j)));
                }
            }
            // https://chat.openai.com/share/7ced3fc8-275a-4134-978c-c822275c3e1f
            // is there a better way to check for within-expected bounds?
            var expectedDelta = vsf == VectorSimilarityFunction.EUCLIDEAN
                    ? 96.98 * log(3.26 + dimension) / log(1.92 + codebooks) - 112.15
                    : 152.69 * log(3.76 + dimension) / log(1.95 + codebooks) - 180.86;
            // expected is accurate to within about 10% *on average*.  experimentally 25% is not quite enough
            // to avoid false positives, so we pad by 40%
            assert delta <= 1.4 * expectedDelta : String.format("%s > %s for %s with %d dimensions and %d codebooks", delta, expectedDelta, vsf, dimension, codebooks);
        }
    }

    @Test
    public void testEncodings() {
        // start with i=2 (dimension 4) b/c dimension 2 is an outlier for our error prediction
        for (int i = 2; i <= 8; i++) {
            for (int M = 1; M <= i; M++) {
                testEncodings(2 * i, M);
            }
        }
    }

    @Test
    public void testCenteringDisturbance() {

    }
}
