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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.AtomicFixedBitSet;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExceptionUtils;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntArrayList;
import org.agrona.collections.IntArrayQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.graph.OnHeapGraphIndex.NO_ENTRY_POINT;
import static io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;
import static io.github.jbellis.jvector.vector.VectorUtil.dotProduct;

/**
 * Builder for Concurrent GraphIndex. See {@link GraphIndex} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 * <p>
 * GIB allocates scratch space and copies of the RandomAccessVectorValues for each thread
 * that calls `addGraphNode`.  These allocations are retained until the GIB itself is no longer referenced.
 * Under most conditions this is not something you need to worry about, but it does mean
 * that spawning a new Thread per call is not advisable.  This includes virtual threads.
 */
public class GraphIndexBuilder implements Closeable {
    private static final Logger logger = LoggerFactory.getLogger(GraphIndexBuilder.class);

    private final int beamWidth;
    private final ExplicitThreadLocal<NodeArray> naturalScratch;
    private final ExplicitThreadLocal<NodeArray> concurrentScratch;

    private final int dimension;
    private final float neighborOverflow;
    private final float alpha;

    @VisibleForTesting
    final OnHeapGraphIndex graph;
    private double averageShortEdges = Double.NaN;

    private final ConcurrentSkipListSet<Integer> insertionsInProgress = new ConcurrentSkipListSet<>();

    private final BuildScoreProvider scoreProvider;

    private final ForkJoinPool simdExecutor;
    private final ForkJoinPool parallelExecutor;

    private final ExplicitThreadLocal<GraphSearcher> searchers;

    private final AtomicInteger updateEntryNodeIn = new AtomicInteger(10_000);

    /**
     * Reads all the vectors from vector values, builds a graph connecting them by their dense
     * ordinals, using the given hyperparameter settings, and returns the resulting graph.
     *
     * @param vectorValues     the vectors whose relations are represented by the graph - must provide a
     *                         different view over those vectors than the one used to add via addGraphNode.
     * @param M                – the maximum number of connections a node can have
     * @param beamWidth        the size of the beam search to use when finding nearest neighbors.
     * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
     *                         node. larger values will build more efficiently, but use more memory.
     * @param alpha            how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
     *                         allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
     *                         an HNSW graph will be created, which is usually not what you want.
     */
    public GraphIndexBuilder(RandomAccessVectorValues vectorValues,
                             VectorSimilarityFunction similarityFunction,
                             int M,
                             int beamWidth,
                             float neighborOverflow,
                             float alpha)
    {
        this(BuildScoreProvider.randomAccessScoreProvider(vectorValues, similarityFunction),
             vectorValues.dimension(),
             M,
             beamWidth,
             neighborOverflow,
             alpha);
    }

    /**
     * Reads all the vectors from vector values, builds a graph connecting them by their dense
     * ordinals, using the given hyperparameter settings, and returns the resulting graph.
     * Default executor pools are used.
     *
     * @param scoreProvider    describes how to determine the similarities between vectors
     * @param M                the maximum number of connections a node can have
     * @param beamWidth        the size of the beam search to use when finding nearest neighbors.
     * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
     *                         node. larger values will build more efficiently, but use more memory.
     * @param alpha            how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
     *                         allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
     *                         an HNSW graph will be created, which is usually not what you want.
     */
    public GraphIndexBuilder(BuildScoreProvider scoreProvider,
                             int dimension,
                             int M,
                             int beamWidth,
                             float neighborOverflow,
                             float alpha)
    {
        this(scoreProvider, dimension, M, beamWidth, neighborOverflow, alpha, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Reads all the vectors from vector values, builds a graph connecting them by their dense
     * ordinals, using the given hyperparameter settings, and returns the resulting graph.
     *
     * @param scoreProvider    describes how to determine the similarities between vectors
     * @param M                the maximum number of connections a node can have
     * @param beamWidth        the size of the beam search to use when finding nearest neighbors.
     * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
     *                         node. larger values will build more efficiently, but use more memory.
     * @param alpha            how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
     *                         allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
     *                         an HNSW graph will be created, which is usually not what you want.
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public GraphIndexBuilder(BuildScoreProvider scoreProvider,
                             int dimension,
                             int M,
                             int beamWidth,
                             float neighborOverflow,
                             float alpha,
                             ForkJoinPool simdExecutor,
                             ForkJoinPool parallelExecutor)
    {
        this.scoreProvider = scoreProvider;
        this.dimension = dimension;
        this.neighborOverflow = neighborOverflow;
        this.alpha = alpha;
        if (M <= 0) {
            throw new IllegalArgumentException("maxConn must be positive");
        }
        if (beamWidth <= 0) {
            throw new IllegalArgumentException("beamWidth must be positive");
        }
        this.beamWidth = beamWidth;
        this.simdExecutor = simdExecutor;
        this.parallelExecutor = parallelExecutor;

        int maxOverflowDegree = (int) (M * neighborOverflow);
        this.graph = new OnHeapGraphIndex(M, maxOverflowDegree, scoreProvider, alpha);
        this.searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(graph));

        // in scratch we store candidates in reverse order: worse candidates are first
        this.naturalScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
        this.concurrentScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
    }

    public OnHeapGraphIndex build(RandomAccessVectorValues ravv) {
        var vv = ravv.threadLocalSupplier();
        int size = ravv.size();

        simdExecutor.submit(() -> {
            IntStream.range(0, size).parallel().forEach(node -> addGraphNode(node, vv.get().getVector(node)));
        }).join();

        cleanup();
        return graph;
    }

    /**
     * Cleanup the graph by completing removal of marked-for-delete nodes, trimming
     * neighbor sets to the advertised degree, and updating the entry node.
     * <p>
     * Uses default threadpool to process nodes in parallel.  There is currently no way to restrict this to a single thread.
     * <p>
     * Must be called before writing to disk.
     * <p>
     * May be called multiple times, but should not be called during concurrent modifications to the graph.
     */
    public void cleanup() {
        if (graph.size() == 0) {
            return;
        }
        graph.validateEntryNode(); // sanity check before we start

        // purge deleted nodes.
        // backlinks can cause neighbors to soft-overflow, so do this before neighbors cleanup
        removeDeletedNodes();

        if (graph.size() == 0) {
            // After removing all the deleted nodes, we might end up with an empty graph.
            // The calls below expect a valid entry node, but we do not have one right now.
            return;
        }

        // clean up overflowed neighbor lists and compute short edges
        averageShortEdges = parallelExecutor.submit(
            () -> IntStream.range(0, graph.getIdUpperBound()).parallel()
                    .mapToDouble(graph.nodes::enforceDegree)
                    .filter(Double::isFinite)
                    .average()
        ).join().orElse(Double.NaN);

        // optimize entry node -- we do this before reconnecting, as otherwise, improving the entry node's
        // connections will tend to disconnect any orphaned nodes reconnected to the entry node
        updateEntryPoint();

        // reconnect any orphaned nodes.  this will maintain neighbors size
        reconnectOrphanedNodes();
    }

    private void reconnectOrphanedNodes() {
        // Set of nodes that may be used as connection targets, initialized to all nodes reachable from the entry
        // point.  But since reconnection edges are usually worse (by distance and/or diversity) than the original
        // ones, we update this as edges are added to avoid reusing the same target node more than once.
        AtomicFixedBitSet globalConnectionTargets = null;
        // Reconnection is best-effort: reconnecting one node may result in disconnecting another, since we are maintaining
        // the maxConnections invariant. So, we do a maximum of 5 loops.
        for (int i = 0; i < 5; i++) {
            // determine the nodes reachable from the entry point at the start of this pass
            var connectedNodes = new AtomicFixedBitSet(graph.getIdUpperBound());
            var entryNeighbors = graph.getNeighbors(graph.entry());
            parallelExecutor.submit(() -> IntStream.range(0, entryNeighbors.size()).parallel().forEach(node -> findConnected(connectedNodes, entryNeighbors.getNode(node)))).join();
            // we deliberately preserve connectionTargets between passes
            if (globalConnectionTargets == null) {
                globalConnectionTargets = connectedNodes.copy();
                // It's particularly important for the entry node to have high quality edges, so mark it
                // as an invalid Target before we start.
                globalConnectionTargets.clear(graph.entry());
            }

            // Gather basic debug information about efficacy/efficiency of reconnection attempts
            var nReconnectAttempts = new AtomicInteger();
            var nReconnectedViaNeighbors = new AtomicInteger();
            var nResumesRun = new AtomicInteger();
            var nReconnectedViaSearch = new AtomicInteger();

            AtomicFixedBitSet connectionTargets = globalConnectionTargets; // effectively final for lambda
            simdExecutor.submit(() -> IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(node -> {
                if (connectedNodes.get(node) || !graph.containsNode(node)) {
                    return;
                }
                nReconnectAttempts.incrementAndGet();

                // first, attempt to connect one of our own connected neighbors to us. Filtering
                // to connectionTargets tends to help for partitioned graphs with large partitions.
                ConcurrentNeighborMap.Neighbors self = graph.getNeighbors(node);
                var neighbors = (NodeArray) self;
                if (connectToClosestNeighbor(node, neighbors, connectionTargets) != null) {
                    nReconnectedViaNeighbors.incrementAndGet();
                    return;
                }

                // if we can't find a connected neighbor to reconnect to, we'll have to search. We start with a small
                // search, and we resume the search in a bounded loop to try to find an eligible connection target.
                // This significantly improves behavior for large (1M+ node) partitioned graphs. We don't add
                // connectionTargets to excludeBits because large partitions lead to excessively large excludeBits,
                // greatly degrading search performance.
                SearchResult result;
                try (var gs = searchers.get()) {
                    var ssp = scoreProvider.searchProviderFor(node);
                    int ep = graph.entry();
                    result = gs.searchInternal(ssp, beamWidth, beamWidth, 0.0f, 0.0f, ep, other -> other != node);
                    neighbors = new NodeArray(result.getNodes().length);
                    toScratchCandidates(result.getNodes(), neighbors);
                    var j = 0;
                    var reconnectedTo = connectToClosestNeighbor(node, neighbors, connectionTargets);
                    // if we can't find a valid connectionTarget within 2*degree of the search destination, give up
                    while (reconnectedTo == null && j < 2 * graph.maxDegree) {
                        j++;
                        nResumesRun.incrementAndGet();
                        result = gs.resume(beamWidth, beamWidth);
                        toScratchCandidates(result.getNodes(), neighbors);
                        reconnectedTo = connectToClosestNeighbor(node, neighbors, connectionTargets);
                    }

                    if (reconnectedTo != null) {
                        nReconnectedViaSearch.incrementAndGet();
                        // since we went to the trouble of finding the closest available neighbor, let `backlink`
                        // check to see if it should be added as an edge to the original node as well
                        var na = new NodeArray(1);
                        na.addInOrder(reconnectedTo.node, reconnectedTo.score);
                        graph.nodes.backlink(na, node, 1.0f);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            })).join();

            logger.debug("Reconnecting {} nodes out of {} on pass {}. {} neighbor reconnects. {} searches/resumes run. {} nodes reconnected via search",
                    nReconnectAttempts.get(), graph.size(), i, nReconnectedViaNeighbors.get(), nResumesRun.get(), nReconnectedViaSearch.get());

            if (nReconnectAttempts.get() == 0) {
                break;
            }
        }
    }

    /**
     * Connect `node` to the closest connected neighbor that is not already a connection target.
     *
     * @return the neighbor id if such a neighbor was found.
     */
    private SearchResult.NodeScore connectToClosestNeighbor(int node, NodeArray neighbors, BitSet connectionTargets) {
        // connect this node to the closest connected neighbor that hasn't already been used as a connection target
        // (since this edge is likely to be the "worst" one in that target's neighborhood, it's likely to be
        // overwritten by the next node to need reconnection if we don't choose a unique target)
        for (int i = 0; i < neighbors.size(); i++) {
            var neighborNode = neighbors.getNode(i);
            if (!connectionTargets.get(neighborNode))
                continue;

            var neighborScore = neighbors.getScore(i);
            graph.nodes.insertEdgeNotDiverse(neighborNode, node, neighborScore);
            connectionTargets.clear(neighborNode);
            return new SearchResult.NodeScore(neighborNode, neighborScore);
        }
        return null;
    }

    private void findConnected(AtomicFixedBitSet connectedNodes, int start) {
        var queue = new IntArrayQueue();
        queue.add(start);
        try (var view = graph.getView()) {
            while (!queue.isEmpty()) {
                // DFS should result in less contention across findConnected threads than BFS
                int next = queue.pollInt();
                if (connectedNodes.getAndSet(next)) {
                    continue;
                }
                for (var it = view.getNeighborsIterator(next); it.hasNext(); ) {
                    queue.addInt(it.nextInt());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public OnHeapGraphIndex getGraph() {
        return graph;
    }

    /**
     * Number of inserts in progress, across all threads.  Useful as a sanity check
     * when calling non-threadsafe methods like cleanup().  (Do not use it to try to
     * _prevent_ races, only to detect them.)
     */
    public int insertsInProgress() {
        return insertionsInProgress.size();
    }

    @Deprecated
    public long addGraphNode(int node, RandomAccessVectorValues ravv) {
        return addGraphNode(node, ravv.getVector(node));
    }

    /**
     * Inserts a node with the given vector value to the graph.
     *
     * <p>To allow correctness under concurrency, we track in-progress updates in a
     * ConcurrentSkipListSet. After adding ourselves, we take a snapshot of this set, and consider all
     * other in-progress updates as neighbor candidates.
     *
     * @param node the node ID to add
     * @return an estimate of the number of extra bytes used by the graph after adding the given node
     */
    public long addGraphNode(int node, VectorFloat<?> vector) {
        // do this before adding to in-progress, so a concurrent writer checking
        // the in-progress set doesn't have to worry about uninitialized neighbor sets
        graph.addNode(node);

        insertionsInProgress.add(node);
        ConcurrentSkipListSet<Integer> inProgressBefore = insertionsInProgress.clone();
        try (var gs = searchers.get()) {
            var naturalScratchPooled = naturalScratch.get();
            var concurrentScratchPooled = concurrentScratch.get();
            // find ANN of the new node by searching the graph
            int ep = graph.entry();

            var bits = new ExcludingBits(node);
            // find best "natural" candidates with a beam search
            var ssp = scoreProvider.searchProviderFor(vector);
            var result = gs.searchInternal(ssp, beamWidth, beamWidth, 0.0f, 0.0f, ep, bits);

            // Update neighbors with these candidates.
            // The DiskANN paper calls for using the entire set of visited nodes along the search path as
            // potential candidates, but in practice we observe neighbor lists being completely filled using
            // just the topK results.  (Since the Robust Prune algorithm prioritizes closer neighbors,
            // this means that considering additional nodes from the search path, that are by definition
            // farther away than the ones in the topK, would not change the result.)
            // TODO if we made NeighborArray an interface we could wrap the NodeScore[] directly instead of copying
            var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
            var concurrent = getConcurrentCandidates(node, inProgressBefore, concurrentScratchPooled, ssp.scoreFunction());
            updateNeighbors(node, natural, concurrent);

            maybeUpdateEntryPoint(node);
            maybeImproveOlderNode();
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            insertionsInProgress.remove(node);
        }

        return graph.ramBytesUsedOneNode();
    }

    /**
     * Improve edge quality on very low-d indexes.  This makes a big difference
     * in the ability of search to escape local maxima to find better options.
     * <p>
     * This has negligible effect on ML embedding-sized vectors, starting at least with GloVe-25, so we don't bother.
     * (Dimensions between 4 and 25 are untested but they get left out too.)
     * For 2D vectors, this takes us to over 99% recall up to at least 4M nodes.  (Higher counts untested.)
    */
    private void maybeImproveOlderNode() {
        // pick a node added earlier at random to improve its connections
        // 20k threshold chosen because that's where recall starts to fall off from 100% for 2D vectors
        if (dimension <= 3 && graph.size() > 20_000) {
            // if we can't find a candidate in 3 tries, the graph is too sparse,
            // we'll have to wait for more nodes to be added (this threshold has been tested w/ parallel build,
            // which generates very sparse ids due to how spliterator works)
            for (int i = 0; i < 3; i++) {
                var olderNode = ThreadLocalRandom.current().nextInt(graph.size());
                if (graph.containsNode(olderNode) && !graph.getDeletedNodes().get(olderNode)) {
                    improveConnections(olderNode);
                    break;
                }
            }
        }
    }

    private void maybeUpdateEntryPoint(int node) {
        graph.maybeSetInitialEntryNode(node); // TODO it seems silly to call this long after we've set it the first time

        if (updateEntryNodeIn.decrementAndGet() == 0) {
            updateEntryPoint();
        }
    }

    @VisibleForTesting
    public void setEntryPoint(int ep) {
        graph.updateEntryNode(ep);
    }

    private void updateEntryPoint() {
        int newEntryNode = approximateMedioid();
        graph.updateEntryNode(newEntryNode);
        if (newEntryNode >= 0) {
            improveConnections(newEntryNode);
            updateEntryNodeIn.addAndGet(graph.size());
        } else {
            updateEntryNodeIn.addAndGet(10_000);
        }
    }

    private void improveConnections(int node) {
        NodeArray naturalScratchPooled;
        SearchResult result;
        try (var gs = searchers.get()) {
            naturalScratchPooled = naturalScratch.get();
            int ep = graph.entry();
            var bits = new ExcludingBits(node);
            var ssp = scoreProvider.searchProviderFor(node);
            result = gs.searchInternal(ssp, beamWidth, beamWidth, 0.0f, 0.0f, ep, bits);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
        var neighbors = graph.nodes.insertDiverse(node, natural);
        // no overflow -- this method gets called from cleanup
        graph.nodes.backlink(neighbors, node, 1.0f);
    }

    public void markNodeDeleted(int node) {
        graph.markDeleted(node);
    }

    /**
     * Remove nodes marked for deletion from the graph, and update neighbor lists
     * to maintain connectivity.  Not threadsafe with respect to other modifications;
     * the `synchronized` flag only prevents concurrent calls to this method.
     *
     * @return approximate size of memory no longer used
     */
    public synchronized long removeDeletedNodes() {
        // Take a snapshot of the nodes to delete
        var toDelete = graph.getDeletedNodes().copy();
        var nRemoved = toDelete.cardinality();
        if (nRemoved == 0) {
            return 0;
        }
        // make a list of remaining live nodes
        var liveNodes = new IntArrayList();
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (graph.containsNode(i) && !toDelete.get(i)) {
                liveNodes.add(i);
            }
        }

        // Compute new edges to insert.  If node j is deleted, we add edges (i, k)
        // whenever (i, j) and (j, k) are directed edges in the current graph.  This
        // strategy is proposed in "FreshDiskANN: A Fast and Accurate Graph-Based
        // ANN Index for Streaming Similarity Search" section 4.2.
        var newEdges = new ConcurrentHashMap<Integer, Set<Integer>>(); // new edges for key k are values v
        parallelExecutor.submit(() -> {
            IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
                var neighbors = graph.getNeighbors(i);
                if (neighbors == null || toDelete.get(i)) {
                    return;
                }
                for (var it = neighbors.iterator(); it.hasNext(); ) {
                    var j = it.nextInt();
                    if (toDelete.get(j)) {
                        var newEdgesForI = newEdges.computeIfAbsent(i, __ -> ConcurrentHashMap.newKeySet());
                        for (var jt = graph.getNeighbors(j).iterator(); jt.hasNext(); ) {
                            int k = jt.nextInt();
                            if (i != k && !toDelete.get(k)) {
                                newEdgesForI.add(k);
                            }
                        }
                    }
                }
            });
        }).join();

        // Remove deleted nodes from neighbors lists;
        // Score the new edges, and connect the most diverse ones as neighbors
        simdExecutor.submit(() -> {
            newEdges.entrySet().stream().parallel().forEach(e -> {
                // turn the new edges into a NodeArray
                int node = e.getKey();
                // each deleted node has ALL of its neighbors added as candidates, so using approximate
                // scoring and then re-scoring only the best options later makes sense here
                var sf = scoreProvider.searchProviderFor(node).scoreFunction();
                var candidates = new NodeArray(graph.maxDegree);
                for (var k : e.getValue()) {
                    candidates.insertSorted(k, sf.similarityTo(k));
                }

                // it's unlikely, but possible, that all the potential replacement edges were to nodes that have also
                // been deleted.  if that happens, keep the graph connected by adding random edges.
                // (this is overly conservative -- really what we care about is that the end result of
                // replaceDeletedNeighbors not be empty -- but we want to avoid having the node temporarily
                // neighborless while concurrent searches run.  empirically, this only results in a little extra work.)
                if (candidates.size() == 0) {
                    var R = ThreadLocalRandom.current();
                    // doing actual sampling-without-replacement is expensive so we'll loop a fixed number of times instead
                    for (int i = 0; i < 2 * graph.maxDegree(); i++) {
                        int randomNode = liveNodes.get(R.nextInt(liveNodes.size()));
                        if (randomNode != node && !candidates.contains(randomNode)) {
                            float score = sf.similarityTo(randomNode);
                            candidates.insertSorted(randomNode, score);
                        }
                        if (candidates.size() == graph.maxDegree) {
                            break;
                        }
                    }
                }

                // remove edges to deleted nodes and add the new connections, maintaining diversity
                graph.nodes.replaceDeletedNeighbors(node, toDelete, candidates);
            });
        }).join();

        // Generally we want to keep entryPoint update and node removal distinct, because both can be expensive,
        // but if the entry point was deleted then we have no choice
        if (toDelete.get(graph.entry())) {
            updateEntryPoint();
        }

        // Remove the deleted nodes from the graph
        assert toDelete.cardinality() == nRemoved : "cardinality changed";
        for (int i = toDelete.nextSetBit(0); i != NO_MORE_DOCS; i = toDelete.nextSetBit(i + 1)) {
            graph.removeNode(i);
        }

        return nRemoved * graph.ramBytesUsedOneNode();
    }

    /**
     * Returns the ordinal of the node that is closest to the centroid of the graph,
     * or NO_ENTRY_POINT if there are no live nodes in the graph.
     */
    private int approximateMedioid() {
        if (graph.size() == 0) {
            return NO_ENTRY_POINT;
        }

        var centroid = scoreProvider.approximateCentroid();
        // if the centroid is the zero vector, pick a random node
        // (this is not a scenario likely to arise outside of small, contrived tests)
        if (dotProduct(centroid, centroid) < 1E-6) {
            return randomLiveNode();
        }

        int ep = graph.entry();
        var ssp = scoreProvider.searchProviderFor(centroid);
        try (var gs = searchers.get()) {
            // search for the centroid.  if we can find a live node nearby, return it
            var result = gs.searchInternal(ssp, beamWidth, beamWidth, 0.0f, 0.0f, ep, Bits.ALL);
            if (result.getNodes().length != 0) {
                return result.getNodes()[0].node;
            }

            // No live nodes found in the search.  Either no live nodes exist, or the graph is too
            // poorly connected to find one.  we'll do our best under the circumstances by picking
            // a random live node, or NO_ENTRY_POINT if none exist.
            return randomLiveNode();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void updateNeighbors(int nodeId, NodeArray natural, NodeArray concurrent) {
        // if either natural or concurrent is empty, skip the merge
        NodeArray toMerge;
        if (concurrent.size() == 0) {
            toMerge = natural;
        } else if (natural.size() == 0) {
            toMerge = concurrent;
        } else {
            toMerge = NodeArray.merge(natural, concurrent);
        }
        // toMerge may be approximate-scored, but insertDiverse will compute exact scores for the diverse ones
        var neighbors = graph.nodes.insertDiverse(nodeId, toMerge);
        graph.nodes.backlink(neighbors, nodeId, neighborOverflow);
    }

    private static NodeArray toScratchCandidates(SearchResult.NodeScore[] candidates, NodeArray scratch) {
        scratch.clear();
        for (var candidate : candidates) {
            scratch.addInOrder(candidate.node, candidate.score);
        }
        return scratch;
    }

    private NodeArray getConcurrentCandidates(int newNode,
                                              Set<Integer> inProgress,
                                              NodeArray scratch,
                                              ScoreFunction scoreFunction)
    {
        scratch.clear();
        for (var n : inProgress) {
            if (n == newNode) {
                continue;
            }
            scratch.insertSorted(n, scoreFunction.similarityTo(n));
        }
        return scratch;
    }

    @Override
    public void close() throws IOException {
        try {
            searchers.close();
        } catch (Exception e) {
            ExceptionUtils.throwIoException(e);
        }
    }

    /**
     * @return a random live node, or `NO_ENTRY_POINT` if no live nodes exist.
     */
    @VisibleForTesting
    int randomLiveNode() {
        var R = ThreadLocalRandom.current();

        // first, try doing it quickly by just picking a random node
        for (int i = 0; i < 3; i++) {
            var idUpperBound = graph.getIdUpperBound();
            if (idUpperBound == 0) {
                return NO_ENTRY_POINT;
            }
            int n = R.nextInt(idUpperBound);
            if (graph.containsNode(n) && !graph.getDeletedNodes().get(n)) {
                return n;
            }
        }

        // lots of deletions and/or sparse node ids, so we do it the slow way
        var L = new ArrayList<Integer>();
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (graph.containsNode(i) && !graph.getDeletedNodes().get(i)) {
                L.add(i);
            }
        }
        if (L.isEmpty()) {
            return NO_ENTRY_POINT;
        }
        return L.get(R.nextInt(L.size()));
    }

    @VisibleForTesting
    void validateAllNodesLive() {
        assert graph.getDeletedNodes().cardinality() == 0;
        // all edges should be valid, live nodes
        for (int i = 0; i < graph.getIdUpperBound(); i++) {
            if (!graph.containsNode(i)) {
                continue; // holes are tolerated
            }
            var neighbors = graph.getNeighbors(i);
            for (var it = neighbors.iterator(); it.hasNext(); ) {
                var j = it.nextInt();
                assert graph.containsNode(j) : String.format("Edge %d -> %d is invalid", i, j);
            }
        }
    }

    /**
     * @return the average short edges.  Will be NaN if cleanup() has not been run,
     * or if no edge lists in the graph needed to be trimmed at cleanup time.
     */
    public double getAverageShortEdges() {
        return averageShortEdges;
    }

    private static class ExcludingBits implements Bits {
        private final int excluded;

        public ExcludingBits(int excluded) {
            this.excluded = excluded;
        }

        @Override
        public boolean get(int index) {
            return index != excluded;
        }
    }

    public void load(RandomAccessReader in) throws IOException {
        if (graph.size() != 0) {
            throw new IllegalStateException("Cannot load into a non-empty graph");
        }

        int size = in.readInt();
        int entryNode = in.readInt();
        int maxDegree = in.readInt();

        for (int i = 0; i < size; i++) {
            int nodeId = in.readInt();
            int nNeighbors = in.readInt();
            var sf = scoreProvider.searchProviderFor(nodeId).exactScoreFunction();
            var ca = new NodeArray(nNeighbors);
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = in.readInt();
                ca.addInOrder(neighbor, sf.similarityTo(neighbor));
            }
            graph.addNode(nodeId, ca);
        }

        graph.updateEntryNode(entryNode);
    }
}
