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
import io.github.jbellis.jvector.graph.GraphIndex.NodeAtLevel;
import io.github.jbellis.jvector.graph.SearchResult.NodeScore;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExceptionUtils;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;
import static java.lang.Math.log;

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

    private final ConcurrentSkipListSet<NodeAtLevel> insertionsInProgress = new ConcurrentSkipListSet<>();

    private final BuildScoreProvider scoreProvider;

    private final ForkJoinPool simdExecutor;
    private final ForkJoinPool parallelExecutor;

    private final ExplicitThreadLocal<GraphSearcher> searchers;

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
        if (M <= 0) {
            throw new IllegalArgumentException("maxConn must be positive");
        }
        if (beamWidth <= 0) {
            throw new IllegalArgumentException("beamWidth must be positive");
        }
        if (neighborOverflow < 1.0f) {
            throw new IllegalArgumentException("neighborOverflow must be >= 1.0");
        }
        if (alpha <= 0) {
            throw new IllegalArgumentException("alpha must be positive");
        }

        this.scoreProvider = scoreProvider;
        this.dimension = dimension;
        this.neighborOverflow = neighborOverflow;
        this.alpha = alpha;
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

    // used by Cassandra when it fine-tunes the PQ codebook
    public static GraphIndexBuilder rescore(GraphIndexBuilder other, BuildScoreProvider newProvider) {
        var newBuilder = new GraphIndexBuilder(newProvider,
                                               other.dimension,
                                               other.graph.maxDegree(),
                                               other.beamWidth,
                                               other.neighborOverflow,
                                               other.alpha,
                                               other.simdExecutor,
                                               other.parallelExecutor);

        // Copy each node and its neighbors from the old graph to the new one
        IntStream.range(0, other.graph.getIdUpperBound()).parallel().forEach(i -> {
            // Find the highest layer this node exists in
            int maxLayer = -1;
            for (int lvl = 0; lvl < other.graph.layers.size(); lvl++) {
                if (other.graph.getNeighbors(lvl, i) == null) {
                    break;
                }
                maxLayer = lvl;
            }
            if (maxLayer < 0) {
                return;
            }

            // Loop over 0..maxLayer, re-score neighbors for each layer
            var sf = newProvider.searchProviderFor(i).scoreFunction();
            for (int lvl = 0; lvl <= maxLayer; lvl++) {
                var oldNeighbors = other.graph.getNeighbors(lvl, i);
                // Copy edges, compute new scores
                var newNeighbors = new NodeArray(oldNeighbors.size());
                for (var it = oldNeighbors.iterator(); it.hasNext();) {
                    int neighbor = it.nextInt();
                    // since we're using a different score provider, use insertSorted instead of addInOrder
                    newNeighbors.insertSorted(neighbor, sf.similarityTo(neighbor));
                }
                newBuilder.graph.addNode(lvl, i, newNeighbors);
            }
        });

        // Set the entry node
        newBuilder.graph.updateEntryNode(other.graph.entry());

        return newBuilder;
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

        // clean up overflowed neighbor lists
        parallelExecutor.submit(() -> {
            IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(id -> {
                for (int layer = 0; layer < graph.layers.size(); layer++) {
                    graph.layers.get(layer).enforceDegree(id);
                }
            });
        }).join();
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

    // This is Lucene's approach, is FAISS's better? MT: It is almost the same, up to a small additive factor
    int getRandomGraphLevel() {
        double ml = graph.maxDegree == 1 ? 1 : 1 / log(1.0 * graph.maxDegree);
        double randDouble;
        do {
            randDouble =
                    ThreadLocalRandom.current().nextDouble(); // avoid 0 value, as log(0) is undefined
        } while (randDouble == 0.0);
        return ((int) (-log(randDouble) * ml));
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
        var nodeLevel = new NodeAtLevel(getRandomGraphLevel(), node);
        graph.addNode(nodeLevel);

        insertionsInProgress.add(nodeLevel);
        var inProgressBefore = insertionsInProgress.clone();
        try (var gs = searchers.get()) {
            gs.setView(graph.getView()); // new snapshot
            var naturalScratchPooled = naturalScratch.get();
            var concurrentScratchPooled = concurrentScratch.get();

            var bits = new ExcludingBits(node);
            var ssp = scoreProvider.searchProviderFor(vector);
            var entry = graph.entry();
            SearchResult result;
            if (entry == null) {
                result = new SearchResult(new NodeScore[] {}, 0, 0, 0);
            } else {
                gs.initializeInternal(ssp, entry, bits);

                // Move downward from entry.level to 1
                for (int lvl = entry.level; lvl > 0; lvl--) {
                    gs.searchOneLayer(ssp, beamWidth, 0.0f, lvl);
                    if (lvl <= nodeLevel.level) {
                        NodeScore[] neighbors = new NodeScore[gs.approximateResults.size()];
                        AtomicInteger index = new AtomicInteger();
                        // TODO extract an interface that lets us avoid the copy here and in toScratchCandidates
                        gs.approximateResults.foreach((neighbor, score) -> {
                            neighbors[index.getAndIncrement()] = new NodeScore(neighbor, score);
                        });
                        Arrays.sort(neighbors);
                        updateNeighborsOneLayer(lvl, node, neighbors, naturalScratchPooled, inProgressBefore, concurrentScratchPooled, ssp);
                    }
                    gs.setEntryPointsFromPreviousLayer();
                }

                // Now do the main search at layer 0
                result = gs.resume(0, beamWidth, beamWidth, 0.0f, 0.0f);
            }

            updateNeighborsOneLayer(0, node, result.getNodes(), naturalScratchPooled, inProgressBefore, concurrentScratchPooled, ssp);

            graph.markComplete(nodeLevel);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            insertionsInProgress.remove(nodeLevel);
        }

        return graph.ramBytesUsedOneNode();
    }

    private void updateNeighborsOneLayer(int layer, int node, NodeScore[] neighbors, NodeArray naturalScratchPooled, ConcurrentSkipListSet<NodeAtLevel> inProgressBefore, NodeArray concurrentScratchPooled, SearchScoreProvider ssp) {
        // Update neighbors with these candidates.
        // The DiskANN paper calls for using the entire set of visited nodes along the search path as
        // potential candidates, but in practice we observe neighbor lists being completely filled using
        // just the topK results.  (Since the Robust Prune algorithm prioritizes closer neighbors,
        // this means that considering additional nodes from the search path, that are by definition
        // farther away than the ones in the topK, would not change the result.)
        var natural = toScratchCandidates(neighbors, naturalScratchPooled);
        var concurrent = getConcurrentCandidates(layer, node, inProgressBefore, concurrentScratchPooled, ssp.scoreFunction());
        updateNeighbors(layer, node, natural, concurrent);
    }

    @VisibleForTesting
    public void setEntryPoint(int level, int node) {
        graph.updateEntryNode(new NodeAtLevel(level, node));
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

        for (int currentLayer = 0; currentLayer < graph.layers.size(); currentLayer++) {
            final int layer = currentLayer;  // Create effectively final copy for lambda
            // Compute new edges to insert.  If node j is deleted, we add edges (i, k)
            // whenever (i, j) and (j, k) are directed edges in the current graph.  This
            // strategy is proposed in "FreshDiskANN: A Fast and Accurate Graph-Based
            // ANN Index for Streaming Similarity Search" section 4.2.
            var newEdges = new ConcurrentHashMap<Integer, Set<Integer>>(); // new edges for key k are values v
            parallelExecutor.submit(() -> {
                IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
                    var neighbors = graph.getNeighbors(layer, i);
                    if (neighbors == null || toDelete.get(i)) {
                        return;
                    }
                    for (var it = neighbors.iterator(); it.hasNext(); ) {
                        var j = it.nextInt();
                        if (toDelete.get(j)) {
                            var newEdgesForI = newEdges.computeIfAbsent(i, __ -> ConcurrentHashMap.newKeySet());
                            for (var jt = graph.getNeighbors(layer, j).iterator(); jt.hasNext(); ) {
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
                    graph.layers.get(layer).replaceDeletedNeighbors(node, toDelete, candidates);
                });
            }).join();
        }

        // Generally we want to keep entryPoint update and node removal distinct, because both can be expensive,
        // but if the entry point was deleted then we have no choice
        if (toDelete.get(graph.entry().node)) {
            // pick a random node at the top layer
            int newLevel = graph.getMaxLevel();
            int newEntry = -1;
            outer:
            while (newLevel > 0) {
                for (var it = graph.getNodes(newLevel); it.hasNext(); ){
                    int i = it.nextInt();
                    if (!toDelete.get(i)) {
                        newEntry = i;
                        break outer;
                    }
                }
                newLevel--;
            }

            graph.updateEntryNode(newEntry >= 0 ? new NodeAtLevel(newLevel, newEntry) : null);
        }

        // Remove the deleted nodes from the graph
        assert toDelete.cardinality() == nRemoved : "cardinality changed";
        for (int i = toDelete.nextSetBit(0); i != NO_MORE_DOCS; i = toDelete.nextSetBit(i + 1)) {
            graph.removeNode(i);
        }

        return nRemoved * graph.ramBytesUsedOneNode();
    }

    private void updateNeighbors(int layer, int nodeId, NodeArray natural, NodeArray concurrent) {
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
        var neighbors = graph.layers.get(layer).insertDiverse(nodeId, toMerge);
        graph.layers.get(layer).backlink(neighbors, nodeId, neighborOverflow);
    }

    private static NodeArray toScratchCandidates(NodeScore[] candidates, NodeArray scratch) {
        scratch.clear();
        for (var candidate : candidates) {
            scratch.addInOrder(candidate.node, candidate.score);
        }
        return scratch;
    }

    private NodeArray getConcurrentCandidates(int layer,
                                              int newNode,
                                              Set<NodeAtLevel> inProgress,
                                              NodeArray scratch,
                                              ScoreFunction scoreFunction)
    {
        scratch.clear();
        for (NodeAtLevel n : inProgress) {
            if (n.node == newNode || n.level < layer) {
                continue;
            }
            scratch.insertSorted(n.node, scoreFunction.similarityTo(n.node));
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
        // TODO
    }
}
