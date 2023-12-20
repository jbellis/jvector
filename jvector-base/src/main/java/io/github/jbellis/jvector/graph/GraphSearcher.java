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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableBitSet;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.util.SparseFixedBitSet;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;

import static java.lang.Math.min;


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
    private final NodeQueue candidates;

    private final BitSet visited;

    /**
     * Creates a new graph searcher.
     *
     * @param visited bit set that will track nodes that have already been visited
     */
    GraphSearcher(GraphIndex.View<T> view, BitSet visited) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.visited = visited;
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static <T> SearchResult search(T targetVector, int topK, RandomAccessVectorValues<T> vectors, VectorEncoding vectorEncoding, VectorSimilarityFunction similarityFunction, GraphIndex<T> graph, Bits acceptOrds) {
        var searcher = new GraphSearcher.Builder<>(graph.getView()).withConcurrentUpdates().build();
        NodeSimilarity.ExactScoreFunction scoreFunction = i -> {
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
     * @param scoreFunction   a function returning the similarity of a given node to the query vector
     * @param reRanker        if scoreFunction is approximate, this should be non-null and perform exact
     *                        comparisons of the vectors for re-ranking at the end of the search.
     * @param topK            the number of results to look for
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. (Experimental!)
     * @param similarityFloor floor for minSimilarity once k results have been considered. (Experimental!)
     *                        This allows early filtering of results when multiple indexes are being searched and we
     *                        are unlikely to consider results below a certain floor. Note that this floor may mean
     *                        navigation of the graph is terminated more quickly, so if we fill topK before the search
     *                        stabilizes, we may see worse results.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    @Experimental
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker<T> reRanker,
                               int topK,
                               float threshold,
                               float similarityFloor,
                               Bits acceptOrds) {
        return searchInternal(scoreFunction, reRanker, topK, threshold, similarityFloor, view.entryNode(), acceptOrds);
    }

    /**
     * @param scoreFunction a function returning the similarity of a given node to the query vector
     * @param reRanker      if scoreFunction is approximate, this should be non-null and perform exact
     *                      comparisons of the vectors for re-ranking at the end of the search.
     * @param topK          the number of results to look for
     * @param threshold     the minimum similarity (0..1) to accept; 0 will accept everything. (Experimental!)
     * @param acceptOrds    a Bits instance indicating which nodes are acceptable results.
     *                      If {@link Bits#ALL}, all nodes are acceptable.
     *                      It is caller's responsibility to ensure that there are enough acceptable nodes
     *                      that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    @Experimental
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker<T> reRanker,
                               int topK,
                               float threshold,
                               Bits acceptOrds) {
        return search(scoreFunction, reRanker, topK, threshold, 0, acceptOrds);
    }


    /**
     * @param scoreFunction a function returning the similarity of a given node to the query vector
     * @param reRanker      if scoreFunction is approximate, this should be non-null and perform exact
     *                      comparisons of the vectors for re-ranking at the end of the search.
     * @param topK          the number of results to look for
     * @param acceptOrds    a Bits instance indicating which nodes are acceptable results.
     *                      If {@link Bits#ALL}, all nodes are acceptable.
     *                      It is caller's responsibility to ensure that there are enough acceptable nodes
     *                      that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker<T> reRanker,
                               int topK,
                               Bits acceptOrds)
    {
        return search(scoreFunction, reRanker, topK, 0.0f, acceptOrds);
    }

    SearchResult searchInternal(NodeSimilarity.ScoreFunction scoreFunction,
                                NodeSimilarity.ReRanker<T> reRanker,
                                int topK,
                                float threshold,
                                int ep,
                                Bits acceptOrds)
    {
        return searchInternal(scoreFunction, reRanker, topK, threshold, 0, ep, acceptOrds);
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
    SearchResult searchInternal(NodeSimilarity.ScoreFunction scoreFunction,
                                NodeSimilarity.ReRanker<T> reRanker,
                                int topK,
                                float threshold,
                                float similarityFloor,
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
        var scoreTracker = threshold > 0 ? new ScoreTracker.NormalDistributionTracker(threshold) : new ScoreTracker.NoOpTracker();
        if (ep < 0) {
            return new SearchResult(new SearchResult.NodeScore[0], visited, 0);
        }

        acceptOrds = Bits.intersectionOf(acceptOrds, view.liveNodes());

        // Threshold callers (and perhaps others) will be tempted to pass in a huge topK.
        // Let's not allocate a ridiculously large heap up front in that scenario.
        var resultsQueue = new NodeQueue(new BoundedLongHeap(min(1024, topK), topK), NodeQueue.Order.MIN_HEAP);
        Map<Integer, T> vectorsEncountered = scoreFunction.isExact() ? null : new java.util.HashMap<>();
        int numVisited = 0;

        float score = scoreFunction.similarityTo(ep);
        visited.set(ep);
        numVisited++;
        candidates.push(ep, score);

        // A bound that holds the minimum similarity to the query vector that a candidate vector must
        // have to be considered.
        float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;

        while (candidates.size() > 0 && !resultsQueue.incomplete()) {
            // done when best candidate is worse than the worst result so far
            float topCandidateScore = candidates.topScore();
            if (topCandidateScore < minAcceptedSimilarity) {
                break;
            }

            // periodically check whether we're likely to find a node above the threshold in the future
            if (scoreTracker.shouldStop(numVisited)) {
                break;
            }

            // add the top candidate to the resultset
            int topCandidateNode = candidates.pop();
            if (acceptOrds.get(topCandidateNode)
                && topCandidateScore >= threshold
                && resultsQueue.push(topCandidateNode, topCandidateScore))
            {
                if (resultsQueue.size() >= topK) {
                    minAcceptedSimilarity = Math.max(similarityFloor, resultsQueue.topScore());
                    while (resultsQueue.topScore() < minAcceptedSimilarity) {
                        resultsQueue.pop();
                    }
                }
                if (!scoreFunction.isExact()) {
                    vectorsEncountered.put(topCandidateNode, view.getVector(topCandidateNode));
                }
            }

            // add its neighbors to the candidates queue
            for (var it = view.getNeighborsIterator(topCandidateNode); it.hasNext(); ) {
                int friendOrd = it.nextInt();
                if (visited.getAndSet(friendOrd)) {
                    continue;
                }
                numVisited++;

                float friendSimilarity = scoreFunction.similarityTo(friendOrd);
                scoreTracker.track(friendSimilarity);
                if (friendSimilarity >= minAcceptedSimilarity) {
                    candidates.push(friendOrd, friendSimilarity);
                }
            }
        }

        assert resultsQueue.size() <= topK;
        SearchResult.NodeScore[] nodes = extractScores(scoreFunction, reRanker, resultsQueue, vectorsEncountered);
        return new SearchResult(nodes, visited, numVisited);
    }

    private static <T> SearchResult.NodeScore[] extractScores(NodeSimilarity.ScoreFunction sf,
                                                              NodeSimilarity.ReRanker<T> reRanker,
                                                              NodeQueue resultsQueue,
                                                              Map<Integer, T> vectorsEncountered)
    {
        SearchResult.NodeScore[] nodes;
        if (sf.isExact()) {
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
        return nodes;
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
