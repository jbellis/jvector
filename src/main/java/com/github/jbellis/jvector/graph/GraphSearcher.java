/*
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

package com.github.jbellis.jvector.graph;

import com.github.jbellis.jvector.util.BitSet;
import com.github.jbellis.jvector.util.Bits;
import com.github.jbellis.jvector.util.FixedBitSet;
import com.github.jbellis.jvector.util.GrowableBitSet;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import static com.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher {
  private final GraphIndex.View view;

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
      GraphIndex.View view,
      BitSet visited) {
    this.view = view;
    this.candidates = new NeighborQueue(100, true);
    this.visited = visited;
  }

  public static <T> NeighborQueue search(T targetVector, int topK, RandomAccessVectorValues<T> copy, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction, GraphIndex graph, Bits acceptOrds, int maxValue) {
    var searcher = new GraphSearcher.Builder(graph.getView()).build();
    return searcher.search(i1 -> {
      switch (vectorEncoding) {
        case BYTE:
          return similarityFunction.compare((byte[]) targetVector, (byte[]) copy.vectorValue(i1));
        case FLOAT32:
          return similarityFunction.compare((float[]) targetVector, (float[]) copy.vectorValue(i1));
        default:
          throw new RuntimeException("Unsupported vector encoding: " + vectorEncoding);
      }
    }, topK, acceptOrds, maxValue);
  }

  /** Builder */
  public static class Builder {
    private final GraphIndex.View graph;
    private boolean concurrent;

    public Builder(GraphIndex.View graph) {
      this.graph = graph;
    }

    public Builder withConcurrentUpdates() {
      this.concurrent = true;
      return this;
    }

    public GraphSearcher build() {
      BitSet bits = concurrent ? new GrowableBitSet(graph.size()) : new FixedBitSet(graph.size());
      return new GraphSearcher(graph, bits);
    }
  }

  public NeighborQueue search(
      NeighborSimilarity.ScoreFunction scoreFunction,
      int topK,
      Bits acceptOrds,
      int visitedLimit) {
    int initialEp = view.entryNode();
    if (initialEp == -1) {
      return new NeighborQueue(1, false);
    }
    NeighborQueue results;
    results = new NeighborQueue(topK, false);
    searchInternal(scoreFunction, results, topK, view.entryNode(), acceptOrds, visitedLimit);
    return results;
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in REVERSE
   * proximity order -- the most distant neighbor of the topK found, i.e. the one with the lowest
   * score/comparison value, will be at the top of the heap, while the closest neighbor will be the
   * last to be popped.
   */
  void searchInternal(
      NeighborSimilarity.ScoreFunction scoreFunction,
      NeighborQueue results,
      int topK,
      int ep,
      Bits acceptOrds,
      int visitedLimit) {
    assert results.isMinHeap();

    if (ep < 0) {
      return;
    }

    prepareScratchState(view.size());

    int numVisited = 0;
    float score = scoreFunction.apply(ep);
    numVisited++;
    visited.set(ep);
    candidates.add(ep, score);
    if (acceptOrds == null || acceptOrds.get(ep)) {
      results.add(ep, score);
    }

    // A bound that holds the minimum similarity to the query vector that a candidate vector must
    // have to be considered.
    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
    if (results.size() >= topK) {
      minAcceptedSimilarity = results.topScore();
    }
    while (candidates.size() > 0 && !results.incomplete()) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      int topCandidateNode = candidates.pop();
      view.seek(topCandidateNode);
      int friendOrd;
      while ((friendOrd = view.nextNeighbor()) != NO_MORE_DOCS) {
        if (visited.getAndSet(friendOrd)) {
          continue;
        }

        if (numVisited >= visitedLimit) {
          results.markIncomplete();
          break;
        }
        float friendSimilarity = scoreFunction.apply(friendOrd);
        numVisited++;
        if (friendSimilarity >= minAcceptedSimilarity) {
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds == null || acceptOrds.get(friendOrd)) {
            if (results.insertWithOverflow(friendOrd, friendSimilarity) && results.size() >= topK) {
              minAcceptedSimilarity = results.topScore();
            }
          }
        }
      }
    }
    while (results.size() > topK) {
      results.pop();
    }
    results.setVisitedCount(numVisited);
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
