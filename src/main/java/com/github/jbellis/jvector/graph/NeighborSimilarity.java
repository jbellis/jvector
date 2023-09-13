package com.github.jbellis.jvector.graph;

import java.util.Map;

/** Encapsulates comparing node distances for diversity checks. */
public interface NeighborSimilarity {
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
   * building the graph (as part of NeighborSimilarity) or for searching it (used standalone,
   * with a reference to the query vector).
   * <p/>
   * ExactScoreFunction and ApproximateScoreFunction are provided for convenience so they
   * can be defined as a simple lambda.
   */
  interface ScoreFunction {
    boolean isExact();

    float similarityTo(int node2);
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

  interface ReRanker<T>  {
    float similarityTo(int node2, Map<Integer, T> vectors);
  }
}
