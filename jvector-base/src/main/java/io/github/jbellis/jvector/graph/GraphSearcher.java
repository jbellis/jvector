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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import io.github.jbellis.jvector.util.GrowableLongHeap;
import io.github.jbellis.jvector.util.SparseBits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntHashSet;

import java.io.Closeable;
import java.io.IOException;


/**
 * Searches a graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link GraphIndex}.
 */
public class GraphSearcher implements Closeable {
    private final GraphIndex.View view;

    // Scratch data structures that are used in each {@link #searchInternal} call. These can be expensive
    // to allocate, so they're cleared and reused across calls.
    private final NodeQueue candidates;
    private final NodeQueue approximateResults;
    private final NodeQueue rerankedResults;
    private final IntHashSet visited;
    private final NodesUnsorted evictedResults;

    // Search parameters that we save here for use by resume()
    private Bits acceptOrds;
    private SearchScoreProvider scoreProvider;
    private CachingReranker cachingReranker;

    /**
     * Creates a new graph searcher from the given GraphIndex
     */
    public GraphSearcher(GraphIndex graph) {
        this(graph.getView());
    }

    private GraphSearcher(GraphIndex.View view) {
        this.view = view;
        this.candidates = new NodeQueue(new GrowableLongHeap(100), NodeQueue.Order.MAX_HEAP);
        this.evictedResults = new NodesUnsorted(100);
        this.approximateResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.rerankedResults = new NodeQueue(new BoundedLongHeap(100), NodeQueue.Order.MIN_HEAP);
        this.visited = new IntHashSet();
    }

    private void initializeScoreProvider(SearchScoreProvider scoreProvider) {
        this.scoreProvider = scoreProvider;
        if (scoreProvider.reranker() == null) {
            cachingReranker = null;
            return;
        }

        cachingReranker = new CachingReranker(scoreProvider);
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
     * @param rerankK         the number of (approximately-scored) results to rerank before returning the best `topK`.
     * @param threshold       the minimum similarity (0..1) to accept; 0 will accept everything. May be used
     *                        with a large topK to find (approximately) all nodes above the given threshold.
     *                        If threshold > 0 then the search will stop when it is probabilistically unlikely
     *                        to find more nodes above the threshold, even if `topK` results have not yet been found.
     * @param rerankFloor     (Experimental!) Candidates whose approximate similarity is at least this value
     *                        will be reranked with the exact score (which requires loading a high-res vector from disk)
     *                        and included in the final results.  (Potentially leaving fewer than topK entries
     *                        in the results.)  Other candidates will be discarded, but will be potentially
     *                        resurfaced if `resume` is called.  This is intended for use when combining results
     *                        from multiple indexes.
     * @param acceptOrds      a Bits instance indicating which nodes are acceptable results.
     *                        If {@link Bits#ALL}, all nodes are acceptable.
     *                        It is caller's responsibility to ensure that there are enough acceptable nodes
     *                        that we don't search the entire graph trying to satisfy topK.
     * @return a SearchResult containing the topK results and the number of nodes visited during the search.
     */
    @Experimental
    public SearchResult search(SearchScoreProvider scoreProvider,
                               int topK,
                               int rerankK,
                               float threshold,
                               float rerankFloor,
                               Bits acceptOrds) {
        return searchInternal(scoreProvider, topK, rerankK, threshold, rerankFloor, view.entryNode(), acceptOrds);
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
        return search(scoreProvider, topK, topK, threshold, 0.0f, acceptOrds);
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
                                int rerankK,
                                float threshold,
                                float rerankFloor,
                                int ep,
                                Bits rawAcceptOrds)
    {
        if (rawAcceptOrds == null) {
            throw new IllegalArgumentException("Use MatchAllBits to indicate that all ordinals are accepted, instead of null");
        }
        if (rerankK < topK) {
            throw new IllegalArgumentException(String.format("rerankK %d must be >= topK %d", rerankK, topK));
        }

        // save search parameters for potential later resume
        initializeScoreProvider(scoreProvider);
        this.acceptOrds = Bits.intersectionOf(rawAcceptOrds, view.liveNodes());

        // reset the scratch data structures
        evictedResults.clear();
        candidates.clear();
        visited.clear();

        // no entry point -> empty results
        if (ep < 0) {
            return new SearchResult(new SearchResult.NodeScore[0], 0, 0, Float.POSITIVE_INFINITY);
        }

        // kick off the actual search at the entry point
        float score = scoreProvider.scoreFunction().similarityTo(ep);
        visited.add(ep);
        candidates.push(ep, score);
        return resume(1, topK, rerankK, threshold, rerankFloor);
    }

    /**
     * Resume the previous search where it left off and search for the best (new) `topK` neighbors.
     * <p>
     * SearchResult.visitedCount resets with each call to `search` or `resume`.
     */
    // Since Astra / Cassandra's usage drives the design decisions here, it's worth being explicit
    // about how that works and why.
    //
    // Astra breaks logical indexes up across multiple physical OnDiskGraphIndex pieces, one per sstable.
    // Each of these pieces is searched independently, and the results are combined.  To avoid doing
    // more work than necessary, Astra assumes that each physical ODGI will contribute responses
    // to the final result in proportion to its size, and only asks for that many results in the initial
    // search.  If this assumption is incorrect, or if the rows found turn out to be deleted or overwritten
    // by later requests (which will be in a different sstable), Astra wants a lightweight way to resume
    // the search where it was left off to get more results.
    //
    // Because Astra uses a nonlinear overquerying strategy (i.e. rerankK will be larger in proportion to
    // topK for small values of topK than for large), it's especially important to avoid reranking more
    // results than necessary.  Thus, Astra will look at the worstApproximateInTopK value from the first
    // ODGI, and use that as the rerankFloor for the next.  Thus, rerankFloor helps avoid believed-to-be-
    // unnecessary work in the initial search, but if the caller needs to resume() then that belief was
    // incorrect and is discarded, and there is no reason to pass a rerankFloor parameter to resume().
    //
    // Finally: in the majority of cases, the initial search() does suffice.  So while we could add the
    // complexity of caching exact scores from candidates that were ultimately evicted from the results,
    // we expect that to be useful only in a small minority of cases -- particularly since we are using
    // rerankFloor to attempt to avoid doing that work in the first place.
    private SearchResult resume(int initialVisited, int topK, int rerankK, float threshold, float rerankFloor) {
        try {
            assert approximateResults.size() == 0; // should be cleared out by extractScores
            assert rerankedResults.size() == 0; // should be cleared out by extractScores
            approximateResults.setMaxSize(rerankK);
            rerankedResults.setMaxSize(topK);

            int numVisited = initialVisited;
            // A bound that holds the minimum similarity to the query vector that a candidate vector must
            // have to be considered -- will be set to the lowest score in the results queue once the queue is full.
            var minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
            // track scores to predict when we are done with threshold queries
            var scoreTracker = threshold > 0 ? new ScoreTracker.TwoPhaseTracker(threshold) : ScoreTracker.NO_OP;
            VectorFloat<?> similarities = null;

            // add evicted results from the last call back to the candidates
            var previouslyEvicted = evictedResults.size() > 0 ? new SparseBits() : Bits.NONE;
            evictedResults.foreach((node, score) -> {
                candidates.push(node, score);
                ((SparseBits) previouslyEvicted).set(node);
            });
            evictedResults.clear();

            // the main search loop
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
                    if (approximateResults.size() < rerankK) {
                        approximateResults.push(topCandidateNode, topCandidateScore);
                        added = true;
                    } else if (topCandidateScore > approximateResults.topScore()) {
                        int evictedNode = approximateResults.topNode();
                        float evictedScore = approximateResults.topScore();
                        evictedResults.add(evictedNode, evictedScore);
                        approximateResults.push(topCandidateNode, topCandidateScore);
                        added = true;
                    } else {
                        added = false;
                    }

                    // update minAcceptedSimilarity if we've found K results
                    if (added && approximateResults.size() >= rerankK) {
                        minAcceptedSimilarity = approximateResults.topScore();
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

            // rerank results
            assert approximateResults.size() <= rerankK;
            NodeQueue popFromQueue;
            float worstApproximateInTopK;
            int reranked;
            if (cachingReranker == null) {
                // save the worst candidates in evictedResults for potential resume()
                while (approximateResults.size() > topK) {
                    var nScore = approximateResults.topScore();
                    var n = approximateResults.pop();
                    evictedResults.add(n, nScore);
                }

                reranked = 0;
                worstApproximateInTopK = Float.POSITIVE_INFINITY;
                popFromQueue = approximateResults;
            } else {
                int oldReranked = cachingReranker.getRerankCalls();
                worstApproximateInTopK = approximateResults.rerank(topK, cachingReranker, rerankFloor, rerankedResults, evictedResults);
                reranked = cachingReranker.getRerankCalls() - oldReranked;
                approximateResults.clear();
                popFromQueue = rerankedResults;
            }
            // pop the top K results from the results queue, which has the worst candidates at the top
            assert popFromQueue.size() <= topK;
            var nodes = new SearchResult.NodeScore[popFromQueue.size()];
            for (int i = nodes.length - 1; i >= 0; i--) {
                var nScore = popFromQueue.topScore();
                var n = popFromQueue.pop();
                nodes[i] = new SearchResult.NodeScore(n, nScore);
            }
            // that should be everything
            assert popFromQueue.size() == 0;

            return new SearchResult(nodes, numVisited, reranked, worstApproximateInTopK);
        } catch (Throwable t) {
            // clear scratch structures if terminated via throwable, as they may not have been drained
            approximateResults.clear();
            rerankedResults.clear();
            throw t;
        }
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
    public SearchResult resume(int additionalK, int rerankK) {
        return resume(0, additionalK, rerankK, 0.0f, 0.0f);
    }

    @Override
    public void close() throws IOException {
        view.close();
    }

    private static class CachingReranker implements ScoreFunction.Reranker {
        // this cache never gets cleared out (until a new search reinitializes it),
        // but we expect resume() to be called at most a few times so it's fine
        private final Int2ObjectHashMap<Float> cachedScores;
        private final SearchScoreProvider scoreProvider;
        private int rerankCalls;

        public CachingReranker(SearchScoreProvider scoreProvider) {
            this.scoreProvider = scoreProvider;
            cachedScores = new Int2ObjectHashMap<>();
            rerankCalls = 0;
        }

        @Override
        public VectorFloat<?> similarityTo(int[] nodes) {
            throw new UnsupportedOperationException();
        }

        @Override
        public float similarityTo(int node2) {
            if (cachedScores.containsKey(node2)) {
                return cachedScores.get(node2);
            }
            rerankCalls++;
            float score = scoreProvider.reranker().similarityTo(node2);
            cachedScores.put(node2, Float.valueOf(score));
            return score;
        }

        public int getRerankCalls() {
            return rerankCalls;
        }
    }
}
