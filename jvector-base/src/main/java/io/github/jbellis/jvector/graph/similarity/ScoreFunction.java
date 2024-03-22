package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.graph.GraphIndex;
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

    boolean isExact();

    float similarityTo(int node2);

    default boolean supportsEdgeLoadingSimilarity() {
        return false;
    }

    default VectorFloat<?> edgeLoadingSimilarityTo(int node2) {
        throw new UnsupportedOperationException("bulk similarity not supported");
    }

    interface ExactScoreFunction extends ScoreFunction {
        default boolean isExact() {
            return true;
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
