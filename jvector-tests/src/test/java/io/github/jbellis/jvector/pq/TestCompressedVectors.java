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
            System.out.printf("delta for %s is %s for dimension %d and %d codebooks%n", vsf, delta, dimension, codebooks);
        }
    }

    @Test
    public void testEncodings() {
        for (int i = 1; i <= 4; i++) {
            for (int M = 1; M <= i; M++) {
                testEncodings(2 * i, M);
            }
        }
    }

    @Test
    public void testCenteringDisturbance() {

    }
}
