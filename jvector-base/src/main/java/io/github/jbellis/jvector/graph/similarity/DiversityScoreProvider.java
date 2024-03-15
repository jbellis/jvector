package io.github.jbellis.jvector.graph.similarity;

/**
 * Instances of DiversityScoreProvider are expected to cache the vectors needed
 * for its lifetime of a single ConcurrentNeighborSet diversity computation,
 * since diversity computations are done pairwise for each of the potential neighbors.
 */
public interface DiversityScoreProvider {
    /**
     * @return the default score function, possibly approximate
     */
    ScoreFunction scoreFunctionFor(int node1);

    /**
     * @return an exact score function
     */
    ScoreFunction.ExactScoreFunction exactScoreFunctionFor(int node1);
}
