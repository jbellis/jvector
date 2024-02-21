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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

/** Encapsulates comparing node distances. */
public interface NodeSimilarity {
    VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** for one-off comparisons between nodes */
    default float score(int node1, int node2) {
        return scoreProvider(node1).similarityTo(node2);
    }

    /**
     * For when we're going to compare node1 with multiple other nodes. This allows us to skip loading
     * node1's vector (potentially from disk) redundantly for each comparison.
     */
    ScoreFunction scoreProvider(int node1);

    /**
     * Provides an API for encapsulating similarity to another node or vector.  Used both for
     * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
     * with a reference to the query vector).
     * <p>
     * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
     * can be defined as a simple lambda.
     */
    interface ScoreFunction {
        boolean isExact();

        float similarityTo(int node2);

        default boolean supportsBulkSimilarity() {
            return false;
        }

        default VectorFloat<?> bulkSimilarityTo(int node2) {
            throw new UnsupportedOperationException("bulk similarity not supported");
        }
    }

    interface ExactScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return true;
        }

        float similarityTo(int node2);
    }

    interface ApproximateScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return false;
        }

        float similarityTo(int node2);
    }

    interface Reranker {
        /**
         * Rerank the nodes identified by the nodes array, storing the similarities in the results vector.
         *
         * @param nodes   the nodes to calculate similarity for
         * @param results the similarity scores for each node, in the same order as the input nodes.
         *                This should be pre-allocated to the same size as nodes.length.
         */
        void score(int[] nodes, VectorFloat<?> results);

        /**
         * Create a Reranker from a VectorSimilarityFunction, a query vector, and a graph view. This is a convenience
         * method for the common case of scoring a set of nodes based on similarity to a query vector.
         *
         * @param queryVector the query vector
         * @param vsf         the similarity function to use
         * @param view        the graph view used to retrieve vectors from node ids
         * @return
         */
        static Reranker from(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, GraphIndex.View view) {
            return (nodes, results) -> {
                var nodeCount = nodes.length;
                var dimension = queryVector.length();
                var packedVectors = vectorTypeSupport.createFloatVector(nodeCount * dimension);
                for (int i1 = 0; i1 < nodeCount; i1++) {
                    var node = nodes[i1];
                    view.getVectorInto(node, packedVectors, i1 * dimension);
                }
                vsf.compareMulti(queryVector, packedVectors, results);
            };
        }
    }
}
