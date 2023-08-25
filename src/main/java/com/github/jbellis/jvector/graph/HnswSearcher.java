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

import java.io.IOException;
import com.github.jbellis.jvector.util.BitSet;
import com.github.jbellis.jvector.util.Bits;
import com.github.jbellis.jvector.util.FixedBitSet;
import com.github.jbellis.jvector.util.GrowableBitSet;

import static com.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Searches an HNSW graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link HnswGraph}.
 *
 * @param <T> the type of query vector
 */
public class HnswSearcher<T> {
  private final HnswGraph graph;
  private final RandomAccessVectorValues<T> vectors; // TODO we don't need this, just need dimension count
  private final NeighborSimilarity.ScoreFunction scoreFunction;
  // TODO correct API is probably
  // new Builder(graph).build().search(scoreFunction, topK, bits, visitLimit)

  /**
   * Scratch data structures that are used in each {@link #searchLevel} call. These can be expensive
   * to allocate, so they're cleared and reused across calls.
   */
  private final NeighborQueue candidates;

  private BitSet visited;

  /**
   * Creates a new graph searcher.
   *
   * @param scoreFunction the similarity function to compare vectors
   * @param visited bit set that will track nodes that have already been visited
   */
  HnswSearcher(
      HnswGraph graph,
      RandomAccessVectorValues<T> vectors,
      NeighborSimilarity.ScoreFunction scoreFunction,
      BitSet visited) {
    this.graph = graph;
    this.vectors = vectors;
    this.scoreFunction = scoreFunction;
    this.candidates = new NeighborQueue(100, true);
    this.visited = visited;
  }

  /** Builder */
  public static class Builder<T> {
    private final HnswGraph graph;
    private final RandomAccessVectorValues<T> vectors;
    private final NeighborSimilarity.ScoreFunction scoreFunction;
    private boolean concurrent;

    public Builder(HnswGraph graph,
                   RandomAccessVectorValues<T> vectors,
                   NeighborSimilarity.ScoreFunction scoreFunction) {
      this.graph = graph;
      this.vectors = vectors;
      this.scoreFunction = scoreFunction;
    }

    public Builder<T> withConcurrentUpdates() {
      this.concurrent = true;
      return this;
    }

    public HnswSearcher<T> build() {
      BitSet bits = concurrent ? new GrowableBitSet(vectors.size()) : new FixedBitSet(vectors.size());
      return new HnswSearcher<>(graph, vectors, scoreFunction, bits);
    }
  }

  public NeighborQueue search(
      int topK,
      Bits acceptOrds,
      int visitedLimit)
      throws IOException {
    int initialEp = graph.entryNode();
    if (initialEp == -1) {
      return new NeighborQueue(1, true);
    }
    NeighborQueue results;
    results = new NeighborQueue(1, false);
    int[] eps = new int[] {graph.entryNode()};
    int numVisited = 0;
    for (int level = graph.numLevels() - 1; level >= 1; level--) {
      results.clear();
      searchLevel(results, 1, level, eps, null, visitedLimit);

      numVisited += results.visitedCount();
      visitedLimit -= results.visitedCount();

      if (results.incomplete()) {
        results.setVisitedCount(numVisited);
        return results;
      }
      eps[0] = results.pop();
    }
    results = new NeighborQueue(topK, false);
    searchLevel(
        results, topK, 0, eps, acceptOrds, visitedLimit);
    results.setVisitedCount(results.visitedCount() + numVisited);
    return results;
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in REVERSE
   * proximity order -- the most distant neighbor of the topK found, i.e. the one with the lowest
   * score/comparison value, will be at the top of the heap, while the closest neighbor will be the
   * last to be popped.
   */
  void searchLevel(
      NeighborQueue results,
      int topK,
      int level,
      final int[] eps,
      Bits acceptOrds,
      int visitedLimit)
      throws IOException {
    assert results.isMinHeap();

    prepareScratchState(vectors.size());

    int numVisited = 0;
    for (int ep : eps) {
      if (visited.getAndSet(ep) == false) {
        if (numVisited >= visitedLimit) {
          results.markIncomplete();
          break;
        }
        float score = scoreFunction.apply(ep);
        numVisited++;
        candidates.add(ep, score);
        if (acceptOrds == null || acceptOrds.get(ep)) {
          results.add(ep, score);
        }
      }
    }

    // A bound that holds the minimum similarity to the query vector that a candidate vector must
    // have to be considered.
    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
    if (results.size() >= topK) {
      minAcceptedSimilarity = results.topScore();
    }
    while (candidates.size() > 0 && results.incomplete() == false) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      int topCandidateNode = candidates.pop();
      graphSeek(graph, level, topCandidateNode);
      int friendOrd;
      while ((friendOrd = graphNextNeighbor(graph)) != NO_MORE_DOCS) {
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

  /**
   * Seek a specific node in the given graph. The default implementation will just call {@link
   * HnswGraph#seek(int, int)}
   *
   * @throws IOException when seeking the graph
   */
  void graphSeek(HnswGraph graph, int level, int targetNode) throws IOException {
    graph.seek(level, targetNode);
  }

  /**
   * Get the next neighbor from the graph, you must call {@link #graphSeek(HnswGraph, int, int)}
   * before calling this method. The default implementation will just call {@link
   * HnswGraph#nextNeighbor()}
   *
   * @return see {@link HnswGraph#nextNeighbor()}
   * @throws IOException when advance neighbors
   */
  int graphNextNeighbor(HnswGraph graph) throws IOException {
    return graph.nextNeighbor();
  }
}
