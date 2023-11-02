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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.util.*;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;


/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher<T> {
  @VisibleForTesting
  // in TestSearchProbability, 100 is not enough to stay within a 10% error rate, but 200 is
  static final int RECENT_SCORES_TRACKED = 200;

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
    var searcher = new GraphSearcher.Builder<>(graph.getView()).withConcurrentUpdates().build();
    NeighborSimilarity.ExactScoreFunction scoreFunction = i -> {
      switch (vectorEncoding) {
        case BYTE:
          return similarityFunction.compare((byte[]) targetVector, (byte[]) vectors.vectorValue(i));
        case FLOAT32:
          return similarityFunction.compare((float[]) targetVector, (float[]) vectors.vectorValue(i));
        default:
          throw new RuntimeException("Unsupported vector encoding: " + vectorEncoding);
      }
    };
    return searcher.search(scoreFunction, null, topK, acceptOrds);
  }

  /** Builder */
  public static class Builder<T> {
    private final GraphIndex.View<T> view;
    private boolean concurrent;

    public Builder(GraphIndex.View<T> view) {
      this.view = view;
    }

    public Builder<T> withConcurrentUpdates() {
      this.concurrent = true;
      return this;
    }

    public GraphSearcher<T> build() {
      int size = view.getIdUpperBound();
      BitSet bits = concurrent ? new GrowableBitSet(size) : new SparseFixedBitSet(size);
      return new GraphSearcher<>(view, bits);
    }
  }

  /**
   * @param scoreFunction a function returning the similarity of a given node to the query vector
   * @param reRanker if scoreFunction is approximate, this should be non-null and perform exact
   *                 comparisons of the vectors for re-ranking at the end of the search.
   * @param topK the number of results to look for
   * @param threshold the minimum similarity (0..1) to accept
   * @param acceptOrds a Bits instance indicating which nodes are acceptable results.
   *                   If null, all nodes are acceptable.
   *                   It is caller's responsibility to ensure that there are enough acceptable nodes
   *                   that we don't search the entire graph trying to satisfy topK.
   * @return a SearchResult containing the topK results and the number of nodes visited during the search.
   */
  public SearchResult search(
      NeighborSimilarity.ScoreFunction scoreFunction,
      NeighborSimilarity.ReRanker<T> reRanker,
      int topK,
      float threshold,
      Bits acceptOrds)
  {
    return searchInternal(scoreFunction, reRanker, topK, threshold, view.entryNode(), acceptOrds);
  }

  public SearchResult search(
          NeighborSimilarity.ScoreFunction scoreFunction,
          NeighborSimilarity.ReRanker<T> reRanker,
          int topK,
          Bits acceptOrds)
  {
    return search(scoreFunction, reRanker, topK, 0.0f, acceptOrds);
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in
   * proximity order -- the closest neighbor of the topK found, i.e. the one with the highest
   * score/comparison value, will be at the front of the array.
   * <p>
   * If scoreFunction is exact, then reRanker may be null.
   * <p>
   * This method never calls acceptOrds.length(), so the length-free Bits.ALL may be passed in.
   */
  // TODO add back ability to re-use a results structure instead of allocating a new one each time?
  SearchResult searchInternal(
          NeighborSimilarity.ScoreFunction scoreFunction,
          NeighborSimilarity.ReRanker<T> reRanker,
          int topK,
          float threshold,
          int ep,
          Bits acceptOrds)
  {
    if (!scoreFunction.isExact() && reRanker == null) {
      throw new IllegalArgumentException("Either scoreFunction must be exact, or reRanker must not be null");
    }
    if (acceptOrds == null) {
      throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
    }

    prepareScratchState(view.size());
    double[] recentScores = threshold > 0 ? new double[RECENT_SCORES_TRACKED] : null;
    int recentScoreIndex = 0; // circular buffer index
    if (ep < 0) {
      return new SearchResult(new SearchResult.NodeScore[0], visited, 0);
    }

    acceptOrds = Bits.intersectionOf(acceptOrds, view.liveNodes());

    var resultsQueue = new NeighborQueue(topK, false);
    Map<Integer, T> vectorsEncountered = scoreFunction.isExact() ? null : new java.util.HashMap<>();
    int numVisited = 0;

    float score = scoreFunction.similarityTo(ep);
    visited.set(ep);
    numVisited++;
    candidates.add(ep, score);
    if (acceptOrds.get(ep) && score >= threshold) {
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
      if (candidates.topScore() < minAcceptedSimilarity) {
        break;
      }

      // periodically check whether we're likely to find a node above the threshold in the future
      if (threshold > 0 && numVisited >= recentScores.length && numVisited % 100 == 0) {
        double futureProbability = futureProbabilityAboveThreshold(recentScores, threshold);
        if (futureProbability < 0.01) {
          break;
        }
      }

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
        if (threshold > 0) {
          recentScores[recentScoreIndex] = friendSimilarity;
          recentScoreIndex = (recentScoreIndex + 1) % RECENT_SCORES_TRACKED;
        }

        if (friendSimilarity >= minAcceptedSimilarity) {
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds.get(friendOrd) && friendSimilarity >= threshold) {
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

    return new SearchResult(nodes, visited, numVisited);
  }

  /**
   * Return the probability of finding a node above the given threshold in the future,
   * given the similarities observed recently.
   */
  @VisibleForTesting
  static double futureProbabilityAboveThreshold(double[] recentSimilarities, double threshold) {
    // Calculate sample mean and standard deviation
    double sampleMean = StatUtils.mean(recentSimilarities);
    double sampleStd = Math.sqrt(StatUtils.variance(recentSimilarities));

    // Z-score for the threshold
    double zScore = (threshold - sampleMean) / sampleStd;

    // Probability of finding a node above the threshold in the future
    NormalDistribution normalDistribution = new NormalDistribution(sampleMean, sampleStd);
    return 1 - normalDistribution.cumulativeProbability(zScore);
  }

  private void prepareScratchState(int capacity) {
    candidates.clear();
    if (visited.length() < capacity) {
      // this happens during graph construction; otherwise the size of the vector values should
      // be constant, and it will be a SparseFixedBitSet instead of FixedBitSet
      if (!(visited instanceof GrowableBitSet)) {
          throw new IllegalArgumentException(
                  String.format("Unexpected visited type: %s. Encountering this means that the graph changed " +
                                "while being searched, and the Searcher was not built withConcurrentUpdates()",
                                visited.getClass().getName()));
      }
      // else GrowableBitSet knows how to grow itself safely
    }
    visited.clear();
  }

}
