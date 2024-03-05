package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public interface Reranker {
    VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

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
