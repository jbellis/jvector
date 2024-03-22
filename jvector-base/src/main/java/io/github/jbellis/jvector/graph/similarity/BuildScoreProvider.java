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

import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorProvider;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2ObjectHashMap;

import java.io.IOException;
import java.util.function.Supplier;

/**
 * Encapsulates comparing node distances for GraphIndexBuilder.
 * <p>
 * It is frustrating but unavoidable that the implementor must write everything twice:
 * once when we know the node id to compare against, but not its vector, and once
 * when we only know the vector--which may not correspond to an existing node.
 * <p>
 * TODO I *think* that LVQ means we can't just define the former in terms of the latter,
 * because the stored LVQ vectors will be a different dimension from the originals.
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
     * @return the approximate centroid of the known nodes.  This is called every time the graph
     * size doubles, and does not block searches or modifications, so it is okay for it to be O(N).
     */
    VectorFloat<?> approximateCentroid();

    /**
     * Create a search score provider to use *internally* during construction.
     * <p>
     * "Internally" means that this may differ from a typical SSP in that it may use
     * approximate scores *without* reranking.  (Reranking will be done separately
     * by the ConcurrentNeighborSet diversity code.)
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
     */
    SearchScoreProvider.Factory diversityProvider();

    /**
     * Returns a BSP that performs exact score comparisons using the given RandomAccessVectorValues and VectorSimilarityFunction.
     */
    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  Usually it's obvious because you can see the different sources being used
        // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
        // and `vectorsCopy` later on when defining the ScoreFunction for search.
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
                    VectorUtil.addInPlace(centroid, vv.vectorValue(i));
                }
                VectorUtil.scale(centroid, 1.0f / vv.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                var vc = vectorsCopy.get();
                var sf = ScoreFunction.ExactScoreFunction.from(vector, similarityFunction, VectorProvider.from(vc));
                return new SearchScoreProvider(sf, null);
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node1) {
                var v = vectors.get().vectorValue(node1);
                return searchProviderFor(v);
            }

            @Override
            public SearchScoreProvider.Factory diversityProvider() {
                return (int node1) -> {
                    var v = vectors.get().vectorValue(node1);
                    var vc = vectorsCopy.get();
                    var sf = ScoreFunction.ExactScoreFunction.from(v, similarityFunction, VectorProvider.from(vc));
                    return new SearchScoreProvider(sf, null);
                };
            }
        };
    }

    /**
     * Returns a BSP that performs approximate score comparisons using the given CompressedVectors,
     * with reranking performed using full resolutions vectors read from the reader
     */
    static BuildScoreProvider pqBuildScoreProvider(VectorSimilarityFunction vsf,
                                                   VectorProvider vp,
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
                    var v1 = cache.computeIfAbsent(node1, vp::get);
                    var sf = cv.scoreFunctionFor(v1, vsf);

                    var cachingVectorProvider = new VectorProvider(dimension) {
                        @Override
                        public void getInto(int node2, VectorFloat<?> result, int offset) {
                            // getInto is only called by reranking, not diversity code
                            throw new UnsupportedOperationException();
                        }

                        @Override
                        public VectorFloat<?> get(int nodeId) {
                            return cache.computeIfAbsent(nodeId, vp::get);
                        }
                    };
                    var rr = ScoreFunction.ExactScoreFunction.from(v1, vsf, cachingVectorProvider);

                    return new SearchScoreProvider(sf, rr);
                };
            }

            @Override
            public SearchScoreProvider searchProviderFor(int node) {
                return searchProviderFor(vp.get(node));
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