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
import io.github.jbellis.jvector.graph.similarity.Reranker;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.AtomicFixedBitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntArrayQueue;
import org.agrona.collections.IntHashSet;

import java.io.IOException;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;
import static io.github.jbellis.jvector.vector.VectorUtil.dotProduct;
import static java.lang.Math.abs;

/**
 * Builder for Concurrent GraphIndex. See {@link GraphIndex} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 * <p>
 * GIB allocates scratch space and copies of the RandomAccessVectorValues for each thread
 * that calls `addGraphNode`.  These allocations are retained until the GIB itself is no longer referenced.
 * Under most conditions this is not something you need to worry about, but it does mean
 * that spawning a new Thread per call is not advisable.  This includes virtual threads.
 */
public class GraphIndexBuilder {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int beamWidth;
    private final ExplicitThreadLocal<NodeArray> naturalScratch;
    private final ExplicitThreadLocal<NodeArray> concurrentScratch;

    private final int dimension;
    private final float neighborOverflow;
    private final float alpha;
    private final ExplicitThreadLocal<GraphSearcher> graphSearcher;

    @VisibleForTesting
    final OnHeapGraphIndex graph;
    private final ConcurrentSkipListSet<Integer> insertionsInProgress = new ConcurrentSkipListSet<>();

    private final BuildScoreProvider scoreProvider;

    private final ForkJoinPool simdExecutor;
    private final ForkJoinPool parallelExecutor;

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
        this(defaultScoreProvider(vectorValues, similarityFunction),
             vectorValues.dimension(),
             M,
             beamWidth,
             neighborOverflow,
             alpha,
             PhysicalCoreExecutor.pool(),
             ForkJoinPool.commonPool());
    }

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
        this.scoreProvider = Objects.requireNonNull(scoreProvider);
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

        this.graph = new OnHeapGraphIndex(M, (node, m) -> new ConcurrentNeighborSet(node, m, scoreProvider, alpha));
        // this view will never get closed, but it's okay because we know it's an OHGI view, which has a no-op close
        this.graphSearcher = ExplicitThreadLocal.withInitial(() -> new GraphSearcher.Builder(graph.getView()).withConcurrentUpdates().build());

        // in scratch we store candidates in reverse order: worse candidates are first
        this.naturalScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
        this.concurrentScratch = ExplicitThreadLocal.withInitial(() -> new NodeArray(Math.max(beamWidth, M + 1)));
    }

    private static BuildScoreProvider defaultScoreProvider(RandomAccessVectorValues ravv, VectorSimilarityFunction similarityFunction) {
        // We need two sources of vectors in order to perform diversity check comparisons without
        // colliding.  Usually it's obvious because you can see the different sources being used
        // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
        // and `vectorsCopy` later on when defining the ScoreFunction for search.
        var vectors = ravv.threadLocalSupplier();
        var vectorsCopy = ravv.threadLocalSupplier();

        return new BuildScoreProvider() {
            @Override
            public ScoreFunction scoreFunctionFor(int node1) {
                var vc = vectorsCopy.get();
                VectorFloat<?> v1 = vectors.get().vectorValue(node1);
                return (ScoreFunction.ExactScoreFunction) node2 -> similarityFunction.compare(v1, vc.vectorValue(node2));
            }

            @Override
            public Reranker rerankerFor(int node1) {
                VectorFloat<?> v1 = vectors.get().vectorValue(node1);
                return (nodes, results) -> {
                    var nodeCount = nodes.length;
                    for (int i = 0; i < nodeCount; i++) {
                        var node2 = nodes[i];
                        var v2 = vectorsCopy.get().vectorValue(node2);
                        // don't use compareMulti, packing the vectors is way too expensive
                        results.set(i, similarityFunction.compare(v1, v2));
                    }
                };
            }

            @Override
            public VectorFloat<?> approximateCentroid() {
                var centroid = vectorTypeSupport.createFloatVector(ravv.dimension());
                for (int i = 0; i < ravv.size(); i++) {
                    VectorUtil.addInPlace(centroid, ravv.vectorValue(i));
                }
                VectorUtil.scale(centroid, 1.0f / ravv.size());
                return centroid;
            }

            @Override
            public SearchScoreProvider searchProviderFor(VectorFloat<?> vector) {
                var sf = (ScoreFunction.ExactScoreFunction) node -> similarityFunction.compare(vector, ravv.vectorValue(node));
                return new SearchScoreProvider(sf, null);
            }

            @Override
            public VectorFloat<?> vectorAt(int node) {
                return ravv.vectorValue(node);
            }
        };
    }

    public OnHeapGraphIndex build(RandomAccessVectorValues ravv) {
        int size = ravv.size();

        simdExecutor.submit(() -> {
            IntStream.range(0, size).parallel().forEach(node -> addGraphNode(node, ravv.vectorValue(node)));
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
     * May be called multiple times, but should not be called during concurrent modifications to the graph
     * or while executing concurrent searches on the graph.
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

        // clean up overflowed neighbor lists
        parallelExecutor.submit(() -> IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
            var neighbors = graph.getNeighbors(i);
            if (neighbors != null) {
                neighbors.enforceDegree();
            }
        })).join();

        // reconnect any orphaned nodes.  this will maintain neighbors size
        reconnectOrphanedNodes();

        // optimize entry node
        graph.updateEntryNode(approximateMedioid());
        updateEntryNodeIn.set(graph.size()); // in case the user goes on to add more nodes after cleanup()
    }

    private void reconnectOrphanedNodes() {
        var searchPathNeighbors = new ConcurrentHashMap<Integer, NodeArray>();
        // It's possible that reconnecting one node will result in disconnecting another, since we are maintaining
        // the maxConnections invariant.  In an extreme case, reconnecting node X disconnects Y, and reconnecting
        // Y disconnects X again.  So we do a best effort of 3 loops.
        for (int i = 0; i < 3; i++) {
            // find all nodes reachable from the entry node
            var connectedNodes = new AtomicFixedBitSet(graph.getIdUpperBound());
            connectedNodes.set(graph.entry());
            var entryNeighbors = graph.getNeighbors(graph.entry()).getCurrent();
            parallelExecutor.submit(() -> IntStream.range(0, entryNeighbors.size).parallel().forEach(node -> findConnected(connectedNodes, entryNeighbors.node[node]))).join();

            // reconnect unreachable nodes
            var nReconnected = new AtomicInteger();
            var connectionTargets = ConcurrentHashMap.<Integer>newKeySet();
            simdExecutor.submit(() -> IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(node -> {
                if (connectedNodes.get(node) || !graph.containsNode(node)) {
                    return;
                }
                nReconnected.incrementAndGet();

                // first, attempt to connect one of our own neighbors to us
                var neighbors = graph.getNeighbors(node).getCurrent();
                if (connectToClosestNeighbor(node, neighbors, connectionTargets)) {
                    return;
                }

                // no unused candidate found -- search for more neighbors and try again
                neighbors = searchPathNeighbors.get(node);
                if (neighbors == null) {
                    var gs = graphSearcher.get();

                    var notSelfBits = createNotSelfBits(node);
                    var ssp = scoreProvider.searchProviderFor(scoreProvider.vectorAt(node));
                    int ep = graph.entry();
                    var result = gs.searchInternal(ssp, beamWidth, 0.0f, 0.0f, ep, notSelfBits);
                    neighbors = new NodeArray(result.getNodes().length);
                    toScratchCandidates(result.getNodes(), neighbors);
                    searchPathNeighbors.put(node, neighbors);
                }
                connectToClosestNeighbor(node, neighbors, connectionTargets);
            }));
            if (nReconnected.get() == 0) {
                break;
            }
        }
    }

    /**
     * Connect `node` to the closest neighbor that is not already a connection target.
     * @return true if such a neighbor was found.
     */
    private boolean connectToClosestNeighbor(int node, NodeArray neighbors, Set<Integer> connectionTargets) {
        // connect this node to the closest neighbor that hasn't already been used as a connection target
        // (since this edge is likely to be the "worst" one in that target's neighborhood, it's likely to be
        // overwritten by the next node to need reconnection if we don't choose a unique target)
        for (int i = 0; i < neighbors.size; i++) {
            var neighborNode = neighbors.node[i];
            var neighborScore = neighbors.score[i];
            if (connectionTargets.add(neighborNode)) {
                graph.getNeighbors(neighborNode).insertNotDiverse(node, neighborScore);
                return true;
            }
        }
        return false;
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
     * Number of inserts in progress, across all threads.
     */
    public int insertsInProgress() {
        return insertionsInProgress.size();
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
        var newNodeNeighbors = graph.addNode(node);

        insertionsInProgress.add(node);
        ConcurrentSkipListSet<Integer> inProgressBefore = insertionsInProgress.clone();
        try {
            var gs = graphSearcher.get();
            var naturalScratchPooled = naturalScratch.get();
            var concurrentScratchPooled = concurrentScratch.get();
            // find ANN of the new node by searching the graph
            int ep = graph.entry();
            var scoreFunction = scoreProvider.scoreFunctionFor(node);

            var bits = new ExcludingBits(node);
            // find best "natural" candidates with a beam search
            var result = gs.searchInternal(scoreProvider.searchProviderFor(vector), beamWidth, 0.0f, 0.0f, ep, bits);

            // Update neighbors with these candidates.
            // The DiskANN paper calls for using the entire set of visited nodes along the search path as
            // potential candidates, but in practice we observe neighbor lists being completely filled using
            // just the topK results.  (Since the Robust Prune algorithm prioritizes closer neighbors,
            // this means that considering additional nodes from the search path, that are by definition
            // farther away than the ones in the topK, would not change the result.)
            // TODO if we made NeighborArray an interface we could wrap the NodeScore[] directly instead of copying
            var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
            var concurrent = getConcurrentCandidates(node, inProgressBefore, concurrentScratchPooled, scoreFunction);
            updateNeighbors(newNodeNeighbors, natural, concurrent);

            maybeUpdateEntryPoint(node);
            maybeImproveOlderNode();
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
                if (graph.containsNode(olderNode)) {
                    improveConnections(olderNode);
                    break;
                }
            }
        }
    }

    private void maybeUpdateEntryPoint(int node) {
        graph.maybeSetInitialEntryNode(node); // TODO it seems silly to call this long after we've set it the first time

        if (updateEntryNodeIn.decrementAndGet() == 0) {
            int newEntryNode = approximateMedioid();
            graph.updateEntryNode(newEntryNode);
            improveConnections(newEntryNode);
            updateEntryNodeIn.addAndGet(graph.size());
        }
    }

    public void improveConnections(int node) {
        var gs = graphSearcher.get();
        var naturalScratchPooled = naturalScratch.get();
        int ep = graph.entry();
        var bits = new ExcludingBits(node);
        var ssp = scoreProvider.searchProviderFor(scoreProvider.vectorAt(node));
        var result = gs.searchInternal(ssp, beamWidth, 0.0f, 0.0f, ep, bits);
        var natural = toScratchCandidates(result.getNodes(), naturalScratchPooled);
        updateNeighbors(graph.getNeighbors(node), natural, NodeArray.EMPTY);
    }

    public void markNodeDeleted(int node) {
        graph.markDeleted(node);
    }

    /**
     * Remove nodes marked for deletion from the graph, and update neighbor lists
     * to maintain connectivity.
     *
     * @return approximate size of memory no longer used
     */
    private long removeDeletedNodes() {
        var deletedNodes = graph.getDeletedNodes();
        var nRemoved = deletedNodes.cardinality();
        if (nRemoved == 0) {
            return 0;
        }

        // remove the nodes from the graph, leaving holes and invalid neighbor references
        for (int i = deletedNodes.nextSetBit(0); i != NO_MORE_DOCS; i = deletedNodes.nextSetBit(i + 1)) {
            var success = graph.removeNode(i);
            assert success : String.format("Node %d marked deleted but not present", i);
        }
        var liveNodes = graph.rawNodes();

        // remove deleted nodes from neighbor lists.  If neighbor count drops below a minimum,
        // add random connections to preserve connectivity
        var affectedLiveNodes = new IntHashSet();
        var R = new Random();
        for (var node : liveNodes) {
            assert !deletedNodes.get(node);

            ConcurrentNeighborSet neighbors = graph.getNeighbors(node);
            if (!neighbors.removeDeletedNeighbors(deletedNodes)) {
                continue;
            }
            affectedLiveNodes.add(node);

            // add random connections if we've dropped below minimum
            var scoreFunction = scoreProvider.scoreFunctionFor(node);
            int minConnections = 1 + graph.maxDegree() / 2;
            if (neighbors.size() < minConnections) {
                // create a NeighborArray of random connections
                NodeArray randomConnections = new NodeArray(graph.maxDegree() - neighbors.size());
                // doing actual sampling-without-replacement is expensive so we'll loop a fixed number of times instead
                for (int i = 0; i < 2 * graph.maxDegree(); i++) {
                    int randomNode = liveNodes[R.nextInt(liveNodes.length)];
                    if (randomNode != node && !randomConnections.contains(randomNode)) {
                        float score = scoreFunction.similarityTo(randomNode);
                        randomConnections.insertSorted(randomNode, score);
                    }
                    if (randomConnections.size == randomConnections.node.length) {
                        break;
                    }
                }
                neighbors.padWith(randomConnections);
            }
        }

        // update entry node if old one was deleted
        if (deletedNodes.get(graph.entry())) {
            if (graph.size() > 0) {
                graph.updateEntryNode(graph.getNodes().nextInt());
            } else {
                graph.updateEntryNode(-1);
            }
        }

        // repair affected nodes
        simdExecutor.submit(() -> affectedLiveNodes.stream().parallel().forEach(this::addNNDescentConnections)).join();

        // reset deleted collection
        deletedNodes.clear();

        return nRemoved * graph.ramBytesUsedOneNode();
    }

    /**
     * Search for the given node, then submit all nodes along the search path as candidates for
     * new neighbors.  Standard diversity pruning applies.
     */
    private void addNNDescentConnections(int node) {
        var notSelfBits = createNotSelfBits(node);
        var gs = graphSearcher.get();
        var scratch = naturalScratch.get();
        int ep = graph.entry();
        var ssp = scoreProvider.searchProviderFor(scoreProvider.vectorAt(node));
        var result = gs.searchInternal(ssp, beamWidth, 0.0f, 0.0f, ep, notSelfBits);
        var candidates = toScratchCandidates(result.getNodes(), scratch);
        updateNeighbors(graph.getNeighbors(node), candidates, NodeArray.EMPTY);
    }

    private static Bits createNotSelfBits(int node) {
        return new Bits() {
            @Override
            public boolean get(int index) {
                return index != node;
            }

            @Override
            public int length() {
                // length is max node id, which could be larger than size after deletes
                throw new UnsupportedOperationException();
            }
        };
    }

    /**
     * Returns the ordinal of the node that is closest to the centroid of the graph.
     */
    private int approximateMedioid() {
        assert graph.size() > 0;

        var gs = graphSearcher.get();
        var centroid = scoreProvider.approximateCentroid();

        // if the centroid is the zero vector, we can't use cosine similarity in our search
        // FIXME
//        VectorSimilarityFunction sf;
//        if (similarityFunction == VectorSimilarityFunction.COSINE) {
//            sf = dotProduct(centroid, centroid) < 1E-6 ? VectorSimilarityFunction.EUCLIDEAN : similarityFunction;
//        } else {
//            sf = similarityFunction;
//        }
        var ssp = scoreProvider.searchProviderFor(centroid);
        int ep = graph.entry();
        var result = gs.searchInternal(ssp, beamWidth, 0.0f, 0.0f, ep, Bits.ALL);
        return result.getNodes()[0].node;
    }

    private void updateNeighbors(ConcurrentNeighborSet neighbors, NodeArray natural, NodeArray concurrent) {
        neighbors.insertDiverse(natural, concurrent);
        neighbors.backlink(graph::getNeighbors, neighborOverflow);
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
                                              NodeArray scratch, ScoreFunction scoreFunction)
    {
        scratch.clear();
        for (var n : inProgress) {
            if (n != newNode) {
                scratch.insertSorted(n, scoreFunction.similarityTo(n));
            }
        }
        return scratch;
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

        @Override
        public int length() {
            throw new UnsupportedOperationException();
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
            int node = in.readInt();
            int nNeighbors = in.readInt();
            var ca = new NodeArray(maxDegree);
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = in.readInt();
                ca.addInOrder(neighbor, scoreProvider.score(node, neighbor));
            }
            graph.addNode(node, new ConcurrentNeighborSet(node, maxDegree, scoreProvider, alpha, ca));
        }

        graph.updateEntryNode(entryNode);
    }
}
