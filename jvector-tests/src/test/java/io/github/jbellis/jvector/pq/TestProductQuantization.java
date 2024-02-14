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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.randomVector;
import static io.github.jbellis.jvector.pq.ProductQuantization.getSubvectorSizesAndOffsets;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Test
    // special cases where each vector maps exactly to a centroid
    public void testPerfectReconstruction() {
        var R = getRandom();

        // exactly the same number of random vectors as clusters
        List<VectorFloat<?>> v1 = IntStream.range(0, ProductQuantization.DEFAULT_CLUSTERS).mapToObj(
                        i -> vectorTypeSupport.createFloatVector(new float[] {R.nextInt(100000), R.nextInt(100000), R.nextInt(100000)}))
                .collect(Collectors.toList());
        assertPerfectQuantization(v1);

        // 10x the number of random vectors as clusters (with duplicates)
        List<VectorFloat<?>> v2 = v1.stream().flatMap(v -> IntStream.range(0, 10).mapToObj(i -> v))
                .collect(Collectors.toList());
        assertPerfectQuantization(v2);
    }

    private static void assertPerfectQuantization(List<VectorFloat<?>> vectors) {
        var ravv = new ListRandomAccessVectorValues(vectors, 3);
        var pq = ProductQuantization.compute(ravv, 2, false);
        var encoded = pq.encodeAll(vectors);
        var decodedScratch = vectorTypeSupport.createFloatVector(3);
        for (int i = 0; i < vectors.size(); i++) {
            pq.decode(encoded[i], decodedScratch);
            assertEquals(vectors.get(i), decodedScratch);
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
        var R = getRandom();
        VectorFloat<?>[] vectors = generate(ProductQuantization.DEFAULT_CLUSTERS + R.nextInt(10*ProductQuantization.DEFAULT_CLUSTERS),
                                     2 + R.nextInt(10),
                                     1_000 + R.nextInt(10_000));

        var clusterer = new KMeansPlusPlusClusterer(vectors, ProductQuantization.DEFAULT_CLUSTERS);
        var initialLoss = loss(clusterer, vectors);

        assert clusterer.clusterOnce() > 0;
        var improvedLoss = loss(clusterer, vectors);

        assertTrue("improvedLoss=" + improvedLoss + " initialLoss=" + initialLoss, improvedLoss < initialLoss);
    }

    @Test
    public void testRefine() {
        var R = getRandom();
        VectorFloat<?>[] vectors = generate(ProductQuantization.DEFAULT_CLUSTERS + R.nextInt(10*ProductQuantization.DEFAULT_CLUSTERS),
                                     2 + R.nextInt(10),
                                     1_000 + R.nextInt(10_000));

        // generate PQ codebooks from half of the dataset
        var half1 = Arrays.copyOf(vectors, vectors.length / 2);
        var ravv1 = new ListRandomAccessVectorValues(List.of(half1), vectors[0].length());
        var pq1 = ProductQuantization.compute(ravv1, 1, false);

        // refine the codebooks with the other half (so, drawn from the same distribution)
        int remaining = vectors.length - vectors.length / 2;
        var half2 = new VectorFloat<?>[remaining];
        System.arraycopy(vectors, vectors.length / 2, half2, 0, remaining);
        var ravv2 = new ListRandomAccessVectorValues(List.of(half2), vectors[0].length());
        var pq2 = pq1.refine(ravv2);

        // the refined version should work better
        var clusterer1 = new KMeansPlusPlusClusterer(half2, pq1.codebooks[0]);
        var clusterer2 = new KMeansPlusPlusClusterer(half2, pq2.codebooks[0]);
        var loss1 = loss(clusterer1, half2);
        var loss2 = loss(clusterer2, half2);
        assertTrue("loss1=" + loss1 + " loss2=" + loss2, loss2 < loss1);
    }


    private static double loss(KMeansPlusPlusClusterer clusterer, VectorFloat<?>[] vectors) {
        var pq = new ProductQuantization(new VectorFloat<?>[] { clusterer.getCentroids() }, ProductQuantization.DEFAULT_CLUSTERS,
                getSubvectorSizesAndOffsets(vectors[0].length(), 1),
                null);

        var encoded = pq.encodeAll(List.of(vectors));
        var loss = 0.0;
        var decodedScratch = vectorTypeSupport.createFloatVector(vectors[0].length());
        for (int i = 0; i < vectors.length; i++) {
            pq.decode(encoded[i], decodedScratch);
            loss += 1 - VectorSimilarityFunction.EUCLIDEAN.compare(vectors[i], decodedScratch);
        }
        return loss;
    }

    private static VectorFloat<?>[] generate(int nClusters, int nDimensions, int nVectors) {
        var R = getRandom();

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
        }).toArray(VectorFloat<?>[]::new);
    }
}
