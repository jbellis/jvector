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

import static java.lang.Math.min;


/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher<T> {

    private final GraphIndex.View<T> view;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    private final BitSet visited;
    // we don't actually need this ordered, but NQ is our only structure that doesn't need to allocate extra containers
    private final NodeQueue evictedResults;

    // Search parameters that we save here for use by resume()
    private NodeSimilarity.ScoreFunction scoreFunction;
    private NodeSimilarity.ReRanker reranker;
    private Bits acceptOrds;

    /**
     * Creates a new graph searcher.
     *
     * @param visited bit set that will track nodes that have already been visited
     */
    GraphSearcher(GraphIndex.View<T> view, BitSet visited) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
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
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param rerankFloor     (Experimental!) Candidates whose approximate similarity is at least this value
     *                        will not be reranked with the exact score (which requires loading the raw vector)
     *                        and included in the final results.  (Potentially leaving fewer than topK entries
     *                        in the results.)  Other candidates will be discarded.  This is intended for use
     *                        when combining results from multiple indexes.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    @Experimental
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker reRanker,
                               int topK,
                               float threshold,
                               float rerankFloor,
                               Bits acceptOrds) {
        return searchInternal(scoreFunction, reRanker, topK, threshold, rerankFloor, view.entryNode(), acceptOrds);
    }

    /**
     * @param scoreFunction   a function returning the similarity of a given node to the query vector
     * @param reRanker        if scoreFunction is approximate, this should be non-null and perform exact
     *                        comparisons of the vectors for re-ranking at the end of the search.
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker reRanker,
                               int topK,
                               float threshold,
                               Bits acceptOrds) {
        return search(scoreFunction, reRanker, topK, threshold, 0.0f, acceptOrds);
    }


    /**
     * @param scoreFunction   a function returning the similarity of a given node to the query vector
     * @param reRanker        if scoreFunction is approximate, this should be non-null and perform exact
     *                        comparisons of the vectors for re-ranking at the end of the search.
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public SearchResult search(NodeSimilarity.ScoreFunction scoreFunction,
                               NodeSimilarity.ReRanker reRanker,
                               int topK,
                               Bits acceptOrds)
    {
        return search(scoreFunction, reRanker, topK, 0.0f, acceptOrds);
    }

    /**
     * Set up the state for a new search and kick it off
     */
    SearchResult searchInternal(NodeSimilarity.ScoreFunction scoreFunction,
                                NodeSimilarity.ReRanker reranker,
                                int topK,
                                float threshold,
                                float rerankFloor,
                                int ep,
                                Bits rawAcceptOrds)
    {
        if (!scoreFunction.isExact() && reranker == null) {
            throw new IllegalArgumentException("Either scoreFunction must be exact, or reranker must not be null");
        }
        if (rawAcceptOrds == null) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }

        this.scoreFunction = scoreFunction;
        this.reranker = reranker;

        prepareScratchState(view.size());
        if (ep < 0) {
            return new SearchResult(new SearchResult.NodeScore[0], visited, 0);
        }

        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        float score = scoreFunction.similarityTo(ep);
        visited.set(ep);
        candidates.push(ep, score);

        var sr = resume(topK, threshold, rerankFloor);
        // include the entry node in visitedCount
        return new SearchResult(sr.getNodes(), sr.getVisited(), sr.getVisitedCount() + 1);
    }

    /**
     * Experimental!
     * <p>
     * Resume the previous search where it left off and search for the best `additionalK` neighbors.
     * It is NOT valid to call this method before calling
     * `search`, but `resume` may be called as many times as desired once the search is initialized.
     * <p>
     * SearchResult.visitedCount resets with each call to `search` or `resume`.
     */
    @Experimental
    public SearchResult resume(int additionalK, float threshold, float rerankFloor) {
        // Threshold callers (and perhaps others) will be tempted to pass in a huge topK.
        // Let's not allocate a ridiculously large heap up front in that scenario.
        var resultsQueue = new NodeQueue(new BoundedLongHeap(min(1024, additionalK), additionalK), NodeQueue.Order.MIN_HEAP);

        int numVisited = 0;
        // A bound that holds the minimum similarity to the query vector that a candidate vector must
        // have to be considered.
        var minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
        var scoreTracker = threshold > 0 ? new ScoreTracker.NormalDistributionTracker(threshold) : ScoreTracker.NO_OP;

        // add evicted results from the last call back to the candidates
        while (evictedResults.size() > 0) {
            float score = evictedResults.topScore();
            int node = evictedResults.pop();
            candidates.push(node, score);
        }
        evictedResults.clear();

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

            // add the top candidate to the resultset if it qualifies, and update minAcceptedSimilarity
            int topCandidateNode = candidates.pop();
            if (acceptOrds.get(topCandidateNode) && topCandidateScore >= threshold) {
                boolean added;
                if (resultsQueue.size() < additionalK) {
                    resultsQueue.push(topCandidateNode, topCandidateScore);
                    added = true;
                } else if (topCandidateScore > resultsQueue.topScore()) {
                    int evictedNode = resultsQueue.topNode();
                    float evictedScore = resultsQueue.topScore();
                    evictedResults.push(evictedNode, evictedScore);
                    resultsQueue.push(topCandidateNode, topCandidateScore);
                    added = true;
                } else {
                    added = false;
                }
                if (added && resultsQueue.size() >= additionalK) {
                    minAcceptedSimilarity = resultsQueue.topScore();
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
                candidates.push(friendOrd, friendSimilarity);
            }
        }

        assert resultsQueue.size() <= additionalK;
        SearchResult.NodeScore[] nodes = extractScores(scoreFunction, reranker, resultsQueue, rerankFloor);
        return new SearchResult(nodes, visited, numVisited);
    }

    /**
     * Experimental!
     * <p>
     * Resume the previous search where it left off and search for the best `additionalK` neighbors.
     * It is NOT valid to call this method before calling
     * `search`, but `resume` may be called as many times as desired once the search is initialized.
     * <p>
     * SearchResult.visitedCount resets with each call to `search` or `resume`.
     */
    @Experimental
    public SearchResult resume(int additionalK) {
        return resume(additionalK, 0.0f, 0.0f);
    }

    /**
     * Empty resultsQueue and rerank its contents, if necessary, and return them in sorted order.
     */
    private static SearchResult.NodeScore[] extractScores(NodeSimilarity.ScoreFunction sf,
                                                          NodeSimilarity.ReRanker reRanker,
                                                          NodeQueue resultsQueue,
                                                          float rerankFloor)
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
            nodes = resultsQueue.nodesCopy(reRanker::similarityTo, rerankFloor);
            Arrays.sort(nodes, 0, nodes.length, Comparator.comparingDouble((SearchResult.NodeScore nodeScore) -> nodeScore.score).reversed());
            resultsQueue.clear();
        }
        return nodes;
    }

    private void prepareScratchState(int capacity) {
        evictedResults.clear();
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
