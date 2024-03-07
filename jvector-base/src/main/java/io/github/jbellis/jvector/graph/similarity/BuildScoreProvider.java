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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/** Encapsulates comparing node distances for GraphIndexBuilder. */
public interface BuildScoreProvider {
    VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * For when we're going to compare node1 with multiple other nodes.  This allows us to skip loading
     * node1's vector (potentially from disk) redundantly for each comparison.
     * <p>
     * Used during searches -- the scoreFunction may be approximate!
     */
    ScoreFunction scoreFunctionFor(int node1);

    /**
     * @return a Reranker that computes exact scores for neighbor candidates.  *Must* be exact!
     */
    Reranker rerankerFor(int node1);

    /**
     * @return true if the score functions returned by this provider are exact.
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
    default SearchScoreProvider searchProviderFor(int node) {
        return new SearchScoreProvider(scoreFunctionFor(node), null);
    }

    static BuildScoreProvider randomAccessScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  Usually it's obvious because you can see the different sources being used
        // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
        // and `vectorsCopy` later on when defining the ScoreFunction for search.
        var vectors = ravv.threadLocalSupplier();
        var vectorsCopy = ravv.threadLocalSupplier();

        return new BuildScoreProvider() {
            @Override
            public ScoreFunction scoreFunctionFor(int node1) {
                var vc = vectorsCopy.get();
                VectorFloat<?> v1 = vectors.get().vectorValue(node1);
                return (ScoreFunction.ExactScoreFunction) node2 -> similarityFunction.compare(v1, vc.vectorValue(node2));
            }

            @Override
            public Reranker rerankerFor(int node1) {
                return null;
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
                var sf = (ScoreFunction.ExactScoreFunction) node -> similarityFunction.compare(vector, vc.vectorValue(node));
                return new SearchScoreProvider(sf, null);
            }

            @Override
            public boolean isExact() {
                return true;
            }
        };
    }
}