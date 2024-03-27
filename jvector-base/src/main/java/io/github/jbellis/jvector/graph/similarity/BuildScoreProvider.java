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
    VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

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
     * @param node the graph node to provide similarity scores against
     */
    SearchScoreProvider searchProviderFor(int node);

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
                var centroid = vectorTypeSupport.createFloatVector(vv.dimension());
                for (int i = 0; i < vv.size(); i++) {
                    VectorUtil.addInPlace(centroid, vv.getVector(i));
                }
                VectorUtil.scale(centroid, 1.0f / vv.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                var vc = vectorsCopy.get();
                var sf = ScoreFunction.ExactScoreFunction.from(vector, similarityFunction, vc);
                return new SearchScoreProvider(sf, null);
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
                    var sf = ScoreFunction.ExactScoreFunction.from(v, similarityFunction, vc);
                    return new SearchScoreProvider(sf, null);
                };
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given PQVectors,
     * with reranking performed using full resolutions vectors read from the reader
     */
    static BuildScoreProvider pqBuildScoreProvider(VectorSimilarityFunction vsf,
                                                   RandomAccessVectorValues vp,
                                                   PQVectors cv)
    {
        int dimension = cv.getOriginalSize() / Float.BYTES;

        return new BuildScoreProvider() {
            @Override
            public boolean isExact() {
                return false;
            }

            @Override
            public SearchScoreProvider.Factory diversityProvider() {
                var cache = new Int2ObjectHashMap<VectorFloat<?>>();
                return node1 -> {
                    var v1 = cache.computeIfAbsent(node1, vp::getVector);
                    var sf = cv.scoreFunctionFor(v1, vsf);

                    var cachedVectors = new RandomAccessVectorValues() {
                        @Override
                        public int size() {
                            return cv.count();
                        }

                        @Override
                        public int dimension() {
                            return dimension;
                        }

                        @Override
                        public boolean isValueShared() {
                            return false;
                        }

                        @Override
                        public RandomAccessVectorValues copy() {
                            return this;
                        }

                        @Override
                        public void getVectorInto(int node2, VectorFloat<?> result, int offset) {
                            // getVectorInto is only called by reranking, not diversity code
                            throw new UnsupportedOperationException();
                        }

                        @Override
                        public VectorFloat<?> getVector(int nodeId) {
                            return cache.computeIfAbsent(nodeId, vp::getVector);
                        }
                    };
                    var rr = ScoreFunction.ExactScoreFunction.from(v1, vsf, cachedVectors);

                    return new SearchScoreProvider(sf, rr);
                };
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node) {
                return searchProviderFor(vp.getVector(node));
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                return new SearchScoreProvider(cv.precomputedScoreFunctionFor(vector, vsf), null);
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                return cv.getCompressor().getOrComputeCentroid();
            }
        };
    }
}