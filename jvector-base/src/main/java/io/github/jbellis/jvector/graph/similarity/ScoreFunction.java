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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/**
 * Provides an API for encapsulating similarity to another node or vector.  Used both for
 * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
 * with a reference to the query vector).
 * <p>
 * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
 * can be defined as a simple lambda.
 */
public interface ScoreFunction {
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * @return true if the ScoreFunction returns exact, full-resolution scores
     */
    boolean isExact();

    /**
     * @return the similarity to one other node
     */
    float similarityTo(int node2);

    /**
     * @return the similarity to all of the nodes that `node2` has an edge towards.
     * Used when expanding the neighbors of a search candidate.
     */
    default VectorFloat<?> edgeLoadingSimilarityTo(int node2) {
        throw new UnsupportedOperationException("bulk similarity not supported");
    }

    /**
     * @return true if `edgeLoadingSimilarityTo` is supported
     */
    default boolean supportsEdgeLoadingSimilarity() {
        return false;
    }

    interface ExactScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return true;
        }

        /**
         * @return a vector of size `nodes.length` containing the corresponding score for
         * each of the nodes in `nodes`.  Used when reranking search results.
         */
        VectorFloat<?> similarityTo(int[] nodes);

        static ExactScoreFunction from(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, RandomAccessVectorValues vp) {
            return new ExactScoreFunction() {
                @Override
                public VectorFloat<?> similarityTo(int[] nodes) {
                    var results = vts.createFloatVector(nodes.length);
                    var nodeCount = nodes.length;
                    var dimension = queryVector.length();
                    var packedVectors = vts.createFloatVector(nodeCount * dimension);
                    for (int i1 = 0; i1 < nodeCount; i1++) {
                        var node = nodes[i1];
                        vp.getVectorInto(node, packedVectors, i1 * dimension);
                    }
                    vsf.compareMulti(queryVector, packedVectors, results);
                    return results;
                }

                @Override
                public float similarityTo(int node2) {
                    return vsf.compare(queryVector, vp.getVector(node2));
                }
            };
        }
    }

    interface ApproximateScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return false;
        }
    }

    interface Provider {
        ScoreFunction scoreFunctionFor(int node1);
    }
}
