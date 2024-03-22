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

/** Encapsulates comparing node distances to a specific vector for GraphSearcher. */
public final class SearchScoreProvider {
    private final ScoreFunction scoreFunction;
    private final ScoreFunction.ExactScoreFunction reranker;

    public SearchScoreProvider(ScoreFunction scoreFunction, ScoreFunction.ExactScoreFunction reranker) {
        this.scoreFunction = scoreFunction;
        this.reranker = reranker;
    }

    public ScoreFunction scoreFunction() {
        return scoreFunction;
    }

    public ScoreFunction.ExactScoreFunction reranker() {
        return reranker;
    }

    public ScoreFunction.ExactScoreFunction exactScoreFunction() {
        return scoreFunction.isExact()
                ? (ScoreFunction.ExactScoreFunction) scoreFunction
                : reranker;
    }
}