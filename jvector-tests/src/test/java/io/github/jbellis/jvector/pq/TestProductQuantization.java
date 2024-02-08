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
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.randomVector;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    @Test
    // special cases where each vector maps exactly to a centroid
    public void testPerfectReconstruction() {
        Random R = getRandom();

        // exactly the same number of random vectors as clusters
        var v1 = IntStream.range(0, ProductQuantization.CLUSTERS).mapToObj(
                i -> new float[] { R.nextInt(100_000), R.nextInt(100_000), R.nextInt(100_000) })
                .collect(Collectors.toList());
        assertPerfectQuantization(v1);

        // 10x the number of random vectors as clusters (with duplicates)
        var v2 = v1.stream().flatMap(v -> IntStream.range(0, 10).mapToObj(i -> v))
                .collect(Collectors.toList());
        assertPerfectQuantization(v2);
    }

    private static void assertPerfectQuantization(List<float[]> vectors) {
        var ravv = new ListRandomAccessVectorValues(vectors, 3);
        var pq = ProductQuantization.compute(ravv, 2, false);
        var encoded = pq.encodeAll(vectors);
        var decodedScratch = new float[3];
        for (int i = 0; i < vectors.size(); i++) {
            pq.decode(encoded[i], decodedScratch);
            assertArrayEquals(Arrays.toString(vectors.get(i)) + "!=" + Arrays.toString(decodedScratch), vectors.get(i), decodedScratch, 0);
        }
    }

    @Test
    // validate that iterating on our cluster centroids improves the encoding
    public void testIterativeImprovement() {
        for (int i = 0; i < 10; i++) {
            testIterativeImprovementOnce();
        }
    }

    public void testIterativeImprovementOnce() {
        Random R = getRandom();
        float[][] vectors = generate(ProductQuantization.CLUSTERS + R.nextInt(10*ProductQuantization.CLUSTERS),
                                     2 + R.nextInt(10),
                                     1_000 + R.nextInt(10_000));

        var clusterer = new KMeansPlusPlusClusterer(vectors, ProductQuantization.CLUSTERS);
        var initialLoss = loss(clusterer, vectors);

        assert clusterer.clusterOnce() > 0;
        var improvedLoss = loss(clusterer, vectors);

        assertTrue(improvedLoss < initialLoss, "improvedLoss=" + improvedLoss + " initialLoss=" + initialLoss);
    }

    @Test
    public void testRefine() {
        Random R = getRandom();
        float[][] vectors = generate(ProductQuantization.CLUSTERS + R.nextInt(10*ProductQuantization.CLUSTERS),
                                     2 + R.nextInt(10),
                                     1_000 + R.nextInt(10_000));

        // generate PQ codebooks from half of the dataset
        var half1 = Arrays.copyOf(vectors, vectors.length / 2);
        var ravv1 = new ListRandomAccessVectorValues(List.of(half1), vectors[0].length);
        var pq1 = ProductQuantization.compute(ravv1, 1, false);

        // refine the codebooks with the other half (so, drawn from the same distribution)
        int remaining = vectors.length - vectors.length / 2;
        var half2 = new float[remaining][];
        System.arraycopy(vectors, vectors.length / 2, half2, 0, remaining);
        var ravv2 = new ListRandomAccessVectorValues(List.of(half2), vectors[0].length);
        var pq2 = pq1.refine(ravv2);

        // the refined version should work better
        var clusterer1 = new KMeansPlusPlusClusterer(half2, pq1.codebooks[0]);
        var clusterer2 = new KMeansPlusPlusClusterer(half2, pq2.codebooks[0]);
        var loss1 = loss(clusterer1, half2);
        var loss2 = loss(clusterer2, half2);
        assertTrue(loss2 < loss1, "loss1=" + loss1 + " loss2=" + loss2);
    }


    private static double loss(KMeansPlusPlusClusterer clusterer, float[][] vectors) {
        var pq = new ProductQuantization(new float[][][] { clusterer.getCentroids() }, null);
        byte[][] encoded = pq.encodeAll(List.of(vectors));

        var decodedScratch = new float[vectors[0].length];
        var loss = 0.0;
        for (int i = 0; i < vectors.length; i++) {
            pq.decode(encoded[i], decodedScratch);
            loss += 1 - VectorSimilarityFunction.COSINE.compare(vectors[i], decodedScratch);
        }
        return loss;
    }

    private static float[][] generate(int nClusters, int nDimensions, int nVectors) {
        Random R = getRandom();

        // generate clusters
        var clusters = IntStream.range(0, nClusters)
                .mapToObj(i -> randomVector(R, nDimensions))
                .collect(Collectors.toList());

        // generate vectors by perturbing clusters
        return IntStream.range(0, nVectors).mapToObj(__ -> {
            var cluster = clusters.get(R.nextInt(nClusters));
            var v = randomVector(R, nDimensions);
            VectorUtil.scale(v, 0.1f + 0.9f * R.nextFloat());
            VectorUtil.addInPlace(v, cluster);
            return v;
        }).toArray(float[][]::new);
    }
}
