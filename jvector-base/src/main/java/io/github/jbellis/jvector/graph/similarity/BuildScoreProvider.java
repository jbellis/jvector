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

package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.CachingVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2ObjectHashMap;

/**
 * Encapsulates comparing node distances for GraphIndexBuilder.
 */
public interface BuildScoreProvider {
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * @return true if the primary score functions used for construction are exact.  This
     * is modestly redundant, but it saves having to allocate new Search/Diversity provider
     * objects in some hot construction loops.
     */
    boolean isExact();

    /**
     * @return the approximate centroid of the known nodes.  We use the closest node
     * to this centroid as the graph entry point, so this is called when the entry point is deleted
     * or every time the graph size doubles.
     * <p>
     * This is not called on a path that blocks searches or modifications, so it is okay for it to be O(N).
     */
    VectorFloat<?> approximateCentroid();

    /**
     * Create a search score provider to use *internally* during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores *without* reranking.  (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * <p>
     * @param vector the query vector to provide similarity scores against
     */
    SearchScoreProvider searchProviderFor(VectorFloat<?> vector);

    /**
     * Create a search score provider to use *internally* during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores *without* reranking.  (In this case, reranking will be done
     * separately by the ConcurrentNeighborSet diversity code.)
     * <p>
     * @param node1 the graph node to provide similarity scores against
     */
    SearchScoreProvider searchProviderFor(int node1);

    /**
     * Create a diversity score provider to use internally during construction.
     * <p>
     * The difference between the diversity provider and the search provider is
     * that the diversity provider MUST include a non-null ExactScoreFunction if the
     * primary score function is approximate.
     * <p>
     * When scoring is approximate, the scores from the search and diversity provider
     * must be consistent, i.e. mixing different types of CompressedVectors will cause problems.
     */
    SearchScoreProvider.Factory diversityProvider();

    /**
     * Returns a BSP that performs exact score comparisons using the given RandomAccessVectorValues and VectorSimilarityFunction.
     */
    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  ThreadLocalSupplier makes this a no-op if the RAVV is actually un-shared.
        var vectors = ravv.threadLocalSupplier();
        var vectorsCopy = ravv.threadLocalSupplier();

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return true;
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                var vv = vectors.get();
                var centroid = vts.createFloatVector(vv.dimension());
                for (int i = 0; i < vv.size(); i++) {
                    var v = vv.getVector(i);
                    if (v != null) { // MapRandomAccessVectorValues is not necessarily dense
                        VectorUtil.addInPlace(centroid, v);
                    }
                }
                VectorUtil.scale(centroid, 1.0f / vv.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                var vc = vectorsCopy.get();
                return SearchScoreProvider.exact(vector, similarityFunction, vc);
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                RandomAccessVectorValues randomAccessVectorValues = vectors.get();
                var v = randomAccessVectorValues.getVector(node1);
                return searchProviderFor(v);
            }

            @Override
            public SearchScoreProvider.Factory diversityProvider() {
                return (int node1) -> {
                    RandomAccessVectorValues randomAccessVectorValues = vectors.get();
                    var v = randomAccessVectorValues.getVector(node1);
                    var vc = vectorsCopy.get();
                    return SearchScoreProvider.exact(v, similarityFunction, vc);
                };
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given PQVectors,
     * with reranking performed using RandomAccessVectorValues (which is intended to be
     * InlineVectorValues or LvqVectorValues for building incrementally, but should technically
     * work with any RAVV implementation).
     */
    static BuildScoreProvider pqBuildScoreProvider(VectorSimilarityFunction vsf,
                                                   RandomAccessVectorValues ravv,
                                                   PQVectors cv)
    {
        int dimension = cv.getOriginalSize() / Float.BYTES;
        assert dimension == ravv.dimension();

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public SearchScoreProvider.Factory diversityProvider() {
                var cache = new Int2ObjectHashMap<VectorFloat<?>>();
                return node1 -> {
                    var cachedVectors = new CachingVectorValues(cv, dimension, cache, ravv);

                    var v1 = cachedVectors.getVector(node1);
                    var asf = cv.scoreFunctionFor(v1, vsf);
                    var rr = ScoreFunction.Reranker.from(v1, vsf, cachedVectors);

                    return new SearchScoreProvider(asf, rr);
                };
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                return searchProviderFor(ravv.getVector(node1));
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                // deliberately skips reranking even though we are using an approximate score function
                return new SearchScoreProvider(cv.precomputedScoreFunctionFor(vector, vsf));
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                return cv.getCompressor().getOrComputeCentroid();
            }
        };
    }

}