package io.github.jbellis.jvector.graph.similarity;

import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Provides an API for encapsulating similarity to another node or vector.  Used both for
 * building the graph (as part of NodeSimilarity) or for searching it (used standalone,
 * with a reference to the query vector).
 * <p>
 * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
 * can be defined as a simple lambda.
 */
public interface ScoreFunction {
    boolean isExact();

    float similarityTo(int node2);

    default boolean supportsBulkSimilarity() {
        return false;
    }

    default VectorFloat<?> bulkSimilarityTo(int node2) {
        throw new UnsupportedOperationException("bulk similarity not supported");
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

    interface Provider {
        ScoreFunction scoreFunctionFor(int node1);
    }
}
