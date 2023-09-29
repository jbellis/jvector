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

import java.util.Map;

/** Encapsulates comparing node distances. */
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
   * <p>
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
