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
import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public final class SearchScoreProvider {
    private final ScoreFunction scoreFunction;
    private final ScoreFunction.Reranker reranker;

    /**
     * @param scoreFunction the primary, fast scoring function
     * @param reranker optional reranking function
     * Generally, reranker will be null iff scoreFunction is an ExactScoreFunction.  However,
     * it is allowed, and sometimes useful, to only perform approximate scoring without reranking.
     * <p>
     * Most often it will be convenient to get the Reranker either using `Reranker.from`
     * or `ScoringView.rerankerFor`.
     */
    public SearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.Reranker reranker) {
        assert scoreFunction != null;
        this.scoreFunction = scoreFunction;
        this.reranker = reranker;
    }

    public ScoreFunction scoreFunction() {
        return scoreFunction;
    }

    public ScoreFunction.Reranker reranker() {
        return reranker;
    }

    public ScoreFunction.ExactScoreFunction exactScoreFunction() {
        return scoreFunction.isExact()
                ? (ScoreFunction.ExactScoreFunction) scoreFunction
                : reranker;
    }

    /**
     * A SearchScoreProvider for a single-pass search based on exact similarity.
     * Generally only suitable when your RandomAccessVectorValues is entirely in-memory,
     * e.g. during construction.
     */
    public static SearchScoreProvider exact(VectorFloat<?> v, VectorSimilarityFunction vsf, RandomAccessVectorValues ravv) {
        var sf = ScoreFunction.Reranker.from(v, vsf, ravv);
        return new SearchScoreProvider(sf, null);
    }

    /**
     * This interface allows implementations to cache the vectors needed
     * for its lifetime of a single ConcurrentNeighborSet diversity computation,
     * since diversity computations are done pairwise for each of the potential neighbors.
     */
    public interface Factory {
        SearchScoreProvider createFor(int node1);
    }
}