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

import io.github.jbellis.jvector.vector.types.VectorFloat;

/** Encapsulates comparing node distances for GraphIndexBuilder. */
public interface BuildScoreProvider {
    /** for one-off comparisons between nodes */
    default float score(int node1, int node2) {
        return scoreFunctionFor(node1).similarityTo(node2);
    }

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
     * @return the approximate centroid of the known nodes.  This is called every time the graph
     * size doubles, and does not block searches or modifications, so it is okay for it to be O(N).
     */
    VectorFloat<?> approximateCentroid();

    SearchScoreProvider searchProviderFor(VectorFloat<?> vector);

    // TODO clean this up, pretty sure we don't need to require vector random access
    VectorFloat<?> vectorAt(int node);
}