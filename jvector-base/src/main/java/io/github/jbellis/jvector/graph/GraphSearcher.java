/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher<T> {
  private final GraphIndex.View<T> view;

  /**
   * Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
   * to allocate, so they're cleared and reused across calls.
   */
  private final NeighborQueue candidates;

  private BitSet visited;

  /**
   * Creates a new graph searcher.
   *
   * @param visited bit set that will track nodes that have already been visited
   */
  GraphSearcher(
      GraphIndex.View<T> view,
      BitSet visited) {
    this.view = view;
    this.candidates = new NeighborQueue(100, true);
    this.visited = visited;
  }

  /**
   * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
   * is the unique owner of the vectors instance passed in here.
   */
  public static <T> SearchResult search(T targetVector, int topK, RandomAccessVectorValues<T> vectors, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction, GraphIndex<T> graph, Bits acceptOrds) {
    var searcher = new GraphSearcher.Builder<>(graph.getView()).build();
    NeighborSimilarity.ExactScoreFunction scoreFunction = i -> {
      switch (vectorEncoding) {
        case BYTE:
          return similarityFunction.compare((VectorByte<?>) targetVector, (VectorByte<?>) vectors.vectorValue(i));
        case FLOAT32:
          return similarityFunction.compare((VectorFloat<?>) targetVector, (VectorFloat<?>) vectors.vectorValue(i));
        default:
          throw new RuntimeException("Unsupported vector encoding: " + vectorEncoding);
      }
    };
    return searcher.search(scoreFunction, null, topK, acceptOrds);
  }

  /** Builder */
  public static class Builder<T> {
    private final GraphIndex.View<T> graph;
    private boolean concurrent;

    public Builder(GraphIndex.View<T> graph) {
      this.graph = graph;
    }

    public Builder<T> withConcurrentUpdates() {
      this.concurrent = true;
      return this;
    }

    public GraphSearcher<T> build() {
      BitSet bits = concurrent ? new GrowableBitSet(graph.size()) : new SparseFixedBitSet(graph.size());
      return new GraphSearcher<>(graph, bits);
    }
  }

  public SearchResult search(
      NeighborSimilarity.ScoreFunction scoreFunction,
      NeighborSimilarity.ReRanker<T> reRanker,
      int topK,
      Bits acceptOrds)
  {
    return searchInternal(scoreFunction, reRanker, topK, view.entryNode(), acceptOrds);
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in
   * proximity order -- the closest neighbor of the topK found, i.e. the one with the highest
   * score/comparison value, will be at the front of the array.
   * <p>
   * If scoreFunction is exact, then reRanker may be null.
   */
  // TODO add back ability to re-use a results structure instead of allocating a new one each time?
  SearchResult searchInternal(
      NeighborSimilarity.ScoreFunction scoreFunction,
      NeighborSimilarity.ReRanker<T> reRanker,
      int topK,
      int ep,
      Bits acceptOrds)
  {
    if (!scoreFunction.isExact() && reRanker == null) {
      throw new IllegalArgumentException("Either scoreFunction must be exact, or reRanker must not be null");
    }

    if (ep < 0) {
      return new SearchResult(new SearchResult.NodeScore[0], 0);
    }

    prepareScratchState(view.size());
    var resultsQueue = new NeighborQueue(topK, false);
    Map<Integer, T> vectorsEncountered = !scoreFunction.isExact() ? new java.util.HashMap<>() : null;
    int numVisited = 0;

    float score = scoreFunction.similarityTo(ep);
    visited.set(ep);
    numVisited++;
    candidates.add(ep, score);
    if (acceptOrds == null || acceptOrds.get(ep)) {
      resultsQueue.add(ep, score);
    }

    // A bound that holds the minimum similarity to the query vector that a candidate vector must
    // have to be considered.
    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
    if (resultsQueue.size() >= topK) {
      minAcceptedSimilarity = resultsQueue.topScore();
    }
    while (candidates.size() > 0 && !resultsQueue.incomplete()) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      // TODO should we merge getVector and getNeighborsIterator into a single method to
      // be more aligned with how it works under the hood?
      int topCandidateNode = candidates.pop();
      if (!scoreFunction.isExact()) {
        vectorsEncountered.put(topCandidateNode, view.getVector(topCandidateNode));
      }
      for (var it = view.getNeighborsIterator(topCandidateNode); it.hasNext(); ) {
        int friendOrd = it.nextInt();
        if (visited.getAndSet(friendOrd)) {
          continue;
        }
        numVisited++;

        float friendSimilarity = scoreFunction.similarityTo(friendOrd);
        if (friendSimilarity >= minAcceptedSimilarity) {
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds == null || acceptOrds.get(friendOrd)) {
            if (resultsQueue.insertWithReplacement(friendOrd, friendSimilarity) && resultsQueue.size() >= topK) {
              minAcceptedSimilarity = resultsQueue.topScore();
            }
          }
        }
      }
    }
    assert resultsQueue.size() <= topK;

    SearchResult.NodeScore[] nodes;
    if (scoreFunction.isExact()) {
      nodes = new SearchResult.NodeScore[resultsQueue.size()];
      for (int i = nodes.length - 1; i >= 0; i--) {
          var nScore = resultsQueue.topScore();
          var n = resultsQueue.pop();
          nodes[i] = new SearchResult.NodeScore(n, nScore);
      }
    } else {
      nodes = resultsQueue.nodesCopy(i -> reRanker.similarityTo(i, vectorsEncountered));
      Arrays.sort(nodes, 0, resultsQueue.size(), Comparator.comparingDouble((SearchResult.NodeScore nodeScore) -> nodeScore.score).reversed());
    }

    return new SearchResult(nodes, numVisited);
  }

  private void prepareScratchState(int capacity) {
    candidates.clear();
    if (visited.length() < capacity) {
      // this happens during graph construction; otherwise the size of the vector values should
      // be constant, and it will be a SparseFixedBitSet instead of FixedBitSet
      assert (visited instanceof FixedBitSet || visited instanceof GrowableBitSet)
          : "Unexpected visited type: " + visited.getClass().getName();
      if (visited instanceof FixedBitSet) {
        visited = FixedBitSet.ensureCapacity((FixedBitSet) visited, capacity);
      }
      // else GrowableBitSet knows how to grow itself safely
    }
    visited.clear();
  }

}
