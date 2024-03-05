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

/** Encapsulates comparing node distances. */
public interface NodeSimilarity {
    /** for one-off comparisons between nodes */
    default float score(int node1, int node2) {
        return scoreProvider(node1).similarityTo(node2);
    }

    /**
     * For when we're going to compare node1 with multiple other nodes. This allows us to skip loading
     * node1's vector (potentially from disk) redundantly for each comparison.
     */
    ScoreFunction scoreProvider(int node1);

}
