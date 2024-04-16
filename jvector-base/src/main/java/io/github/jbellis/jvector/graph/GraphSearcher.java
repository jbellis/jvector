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
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.util.SparseBits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntHashSet;

import java.util.Arrays;
import java.util.Comparator;


/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher implements AutoCloseable {
    private final GraphIndex.View view;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    private final NodeQueue resultsQueue;
    private final IntHashSet visited;
    // we don't actually need this ordered, but NQ is our only structure that doesn't need to allocate extra containers
    private final NodeQueue evictedResults;

    // Search parameters that we save here for use by resume()
    private Bits acceptOrds;
    private SearchScoreProvider scoreProvider;

    /**
     * Creates a new graph searcher from the given GraphIndex
     */
    public GraphSearcher(GraphIndex graph) {
        this(graph.getView());
    }

    private GraphSearcher(GraphIndex.View view) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.resultsQueue = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
    }

    public GraphIndex.View getView() {
        return view;
    }

    /**
     * Convenience function for simple one-off searches.  It is caller's responsibility to make sure that it
     * is the unique owner of the vectors instance passed in here.
     */
    public static SearchResult search(VectorFloat<?> queryVector, int topK, RandomAccessVectorValues vectors, VectorSimilarityFunction similarityFunction, GraphIndex graph, Bits acceptOrds) {
        try (var searcher = new GraphSearcher(graph)) {
            var ssp = SearchScoreProvider.exact(queryVector, similarityFunction, vectors);
            return searcher.search(ssp, topK, acceptOrds);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Call GraphSearcher constructor instead
     */
    @Deprecated
    public static class Builder {
        private final GraphIndex.View view;

        public Builder(GraphIndex.View view) {
            this.view = view;
        }

        public Builder withConcurrentUpdates() {
            return this;
        }

        public GraphSearcher build() {
            return new GraphSearcher(view);
        }
    }

    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
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
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               float threshold,
                               float rerankFloor,
                               Bits acceptOrds) {
        return searchInternal(scoreProvider, topK, threshold, rerankFloor, view.entryNode(), acceptOrds);
    }

    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
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
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               float threshold,
                               Bits acceptOrds) {
        return search(scoreProvider, topK, threshold, 0.0f, acceptOrds);
    }


    /**
     * @param scoreProvider   provides functions to return the similarity of a given node to the query vector
     * @param topK            the number of results to look for. With threshold=0, the search will continue until at least
     *                        `topK` results have been found, or until the entire graph has been searched.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               Bits acceptOrds)
    {
        return search(scoreProvider, topK, 0.0f, acceptOrds);
    }

    /**
     * Set up the state for a new search and kick it off
     */
    SearchResult searchInternal(SearchScoreProvider scoreProvider,
                                int topK,
                                float threshold,
                                float rerankFloor,
                                int ep,
                                Bits rawAcceptOrds)
    {
        if (rawAcceptOrds == null) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }

        // save search parameters for potential later resume
        this.scoreProvider = scoreProvider;
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        // reset the scratch data structures
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        // no entry point -> empty results
        if (ep < 0) {
            return new SearchResult(new SearchResult.NodeScore[0], 0);
        }

        // kick off the actual search at the entry point
        float score = scoreProvider.scoreFunction().similarityTo(ep);
        visited.add(ep);
        candidates.push(ep, score);
        var sr = resume(topK, threshold, rerankFloor);

        // include the entry node in visitedCount
        return new SearchResult(sr.getNodes(), sr.getVisitedCount() + 1);
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
        assert resultsQueue.size() == 0; // should be cleared out by extractScores
        resultsQueue.setMaxSize(additionalK);

        int numVisited = 0;
        // A bound that holds the minimum similarity to the query vector that a candidate vector must
        // have to be considered -- will be set to the lowest score in the results queue once the queue is full.
        var minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
        // track scores to predict when we are done with threshold queries
        var scoreTracker = threshold > 0 ? new ScoreTracker.TwoPhaseTracker(threshold) : ScoreTracker.NO_OP;
        VectorFloat<?> similarities = null;

        // add evicted results from the last call back to the candidates
        var previouslyEvicted = evictedResults.size() > 0 ? new SparseBits() : Bits.NONE;
        while (evictedResults.size() > 0) {
            float score = evictedResults.topScore();
            int node = evictedResults.pop();
            candidates.push(node, score);
            ((SparseBits) previouslyEvicted).set(node);
        }
        evictedResults.clear();

        while (candidates.size() > 0) {
            // we're done when we have K results and the best candidate is worse than the worst result so far
            float topCandidateScore = candidates.topScore();
            if (topCandidateScore < minAcceptedSimilarity) {
                break;
            }
            // when querying by threshold, also stop when we are probabilistically unlikely to find more qualifying results
            if (scoreTracker.shouldStop()) {
                break;
            }

            // process the top candidate
            int topCandidateNode = candidates.pop();
            if (acceptOrds.get(topCandidateNode) && topCandidateScore >= threshold) {
                // add the new node to the results queue, and any evicted node to evictedResults in case we resume later
                // (push() can't tell us what node was evicted when the queue was already full, so we examine that manually)
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

                // update minAcceptedSimilarity if we've found K results
                if (added && resultsQueue.size() >= additionalK) {
                    minAcceptedSimilarity = resultsQueue.topScore();
                }
            }

            // if this candidate came from evictedResults, we don't need to evaluate its neighbors again
            if (previouslyEvicted.get(topCandidateNode)) {
                continue;
            }

            // score the neighbors of the top candidate and add them to the queue
            var scoreFunction = scoreProvider.scoreFunction();
            var useEdgeLoading = scoreFunction.supportsEdgeLoadingSimilarity();
            if (useEdgeLoading) {
                similarities = scoreFunction.edgeLoadingSimilarityTo(topCandidateNode);
            }

            var it = view.getNeighborsIterator(topCandidateNode);
            for (int i = 0; i < it.size(); i++) {
                var friendOrd = it.nextInt();
                if (!visited.add(friendOrd)) {
                    continue;
                }
                numVisited++;

                float friendSimilarity = useEdgeLoading
                        ? similarities.get(i)
                        : scoreFunction.similarityTo(friendOrd);
                scoreTracker.track(friendSimilarity);
                candidates.push(friendOrd, friendSimilarity);
            }
        }

        assert resultsQueue.size() <= additionalK;
        SearchResult.NodeScore[] nodes = extractScores(scoreProvider, resultsQueue, rerankFloor);
        return new SearchResult(nodes, numVisited);
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
    private static SearchResult.NodeScore[] extractScores(SearchScoreProvider scoreProvider,
                                                          NodeQueue resultsQueue,
                                                          float rerankFloor)
    {
        SearchResult.NodeScore[] nodes;
        if (scoreProvider.reranker() == null) {
            nodes = new SearchResult.NodeScore[resultsQueue.size()];
            for (int i = nodes.length - 1; i >= 0; i--) {
                var nScore = resultsQueue.topScore();
                var n = resultsQueue.pop();
                nodes[i] = new SearchResult.NodeScore(n, nScore);
            }
        } else {
            nodes = resultsQueue.nodesCopy(scoreProvider.reranker(), rerankFloor);
            Arrays.sort(nodes, 0, nodes.length, Comparator.comparingDouble((SearchResult.NodeScore nodeScore) -> nodeScore.score).reversed());
            resultsQueue.clear();
        }
        return nodes;
    }

    @Override
    public void close() throws Exception {
        view.close();
    }
}
