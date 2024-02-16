/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Tests KNN graphs
 */
public class TestFloatVectorGraph extends GraphIndexTestCase<float[]> {

    @Before
    public void setup() {
        similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
    }

    @Override
    VectorEncoding getVectorEncoding() {
        return VectorEncoding.FLOAT32;
    }

    @Override
    float[] randomVector(int dim) {
        return TestUtil.randomVector(getRandom(), dim);
    }

    @Override
    AbstractMockVectorValues<float[]> vectorValues(int size, int dimension) {
        return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, getRandom()));
    }

    @Override
    AbstractMockVectorValues<float[]> vectorValues(float[][] values) {
        return MockVectorValues.fromValues(values);
    }

    @Override
    RandomAccessVectorValues<float[]> circularVectorValues(int nDoc) {
        return new CircularFloatVectorValues(nDoc);
    }

    @Override
    float[] getTargetVector() {
        return new float[]{1f, 0f};
    }

    public void testSearchWithSkewedAcceptOrds() {
        int nDoc = 1000;
        similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        RandomAccessVectorValues<float[]> vectors = circularVectorValues(nDoc);
        VectorEncoding vectorEncoding = getVectorEncoding();
        getRandom().nextInt();
        GraphIndexBuilder<float[]> builder = new GraphIndexBuilder<>(vectors, vectorEncoding, similarityFunction, 32, 100, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, vectors);

        // Skip over half of the documents that are closest to the query vector
        FixedBitSet acceptOrds = new FixedBitSet(nDoc);
        for (int i = 500; i < nDoc; i++) {
            acceptOrds.set(i);
        }
        SearchResult.NodeScore[] nn =
                GraphSearcher.search(
                        getTargetVector(),
                        10,
                        vectors.copy(),
                        getVectorEncoding(),
                        similarityFunction,
                        graph,
                        acceptOrds
                ).getNodes();

        int[] nodes = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
        int sum = 0;
        for (int node : nodes) {
            assertTrue("the results include a deleted document: " + node, acceptOrds.get(node));
            sum += node;
        }
        // We still expect to get reasonable recall. The lowest non-skipped docIds
        // are closest to the query vector: sum(500,509) = 5045
        assertTrue("sum(result docs)=" + sum, sum < 5100);
    }

    @Test
    // build a random graph and check that resuming a search finds the same nodes as an equivalent from-search search
    // this test is float-specific because random byte vectors are far more likely to have tied similarities,
    // which throws off our assumption that resume picks back up with the same state that the original search
    // left off in (because evictedResults from the first search may not end up in the same order in the
    // candidates queue)
    public void testResume() {
        int size = 1000;
        int dim = 2;
        var vectors = vectorValues(size, dim);
        var builder = new GraphIndexBuilder<>(vectors, getVectorEncoding(), similarityFunction, 20, 30, 1.0f, 1.4f);
        var graph = builder.build();
        Bits acceptOrds = getRandom().nextBoolean() ? Bits.ALL : createRandomAcceptOrds(0, size);

        int initialTopK = 10;
        int resumeTopK = 15;
        var query = randomVector(dim);
        var searcher = new GraphSearcher.Builder<>(graph.getView()).build();

        var initial = searcher.search(getScoreFunction(query, vectors), null, initialTopK, acceptOrds);
        assertEquals(initialTopK, initial.getNodes().length);

        var resumed = searcher.resume(resumeTopK);
        assertEquals(resumeTopK, resumed.getNodes().length);

        var expected = searcher.search(getScoreFunction(query, vectors), null, initialTopK + resumeTopK, acceptOrds);
        assertEquals(expected.getVisitedCount(), initial.getVisitedCount() + resumed.getVisitedCount());
        assertEquals(expected.getNodes().length, initial.getNodes().length + resumed.getNodes().length);
        var initialResumedResults = Stream.concat(Arrays.stream(initial.getNodes()), Arrays.stream(resumed.getNodes()))
                .sorted(Comparator.comparingDouble(ns -> -ns.score))
                .collect(Collectors.toList());
        var expectedResults = List.of(expected.getNodes());
        for (int i = 0; i < expectedResults.size(); i++) {
            assertEquals(expectedResults.get(i).score, initialResumedResults.get(i).score, 1E-6);
        }
    }

    @Test
    public void testZeroCentroid()
    {
        var rawVectors = List.of(new float[] {-1, -1}, new float[] {1, 1});
        var vectors = new ListRandomAccessVectorValues(rawVectors, 2);
        var builder = new GraphIndexBuilder<>(vectors, getVectorEncoding(), VectorSimilarityFunction.COSINE, 2, 2, 1.0f, 1.0f);
        try (var graph = builder.build()) {
            var results = GraphSearcher.search(new float[] {0.5f, 0.5f}, 1, vectors, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            assertEquals(1, results.getNodes().length);
            assertEquals(1, results.getNodes()[0].node);
        }
    }
}
