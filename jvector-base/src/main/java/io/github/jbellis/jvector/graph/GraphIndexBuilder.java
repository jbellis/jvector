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
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Builder for Concurrent GraphIndex. See {@link GraphIndex} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 *
 * @param <T> the type of vector
 */
public class GraphIndexBuilder<T> {
    private final int beamWidth;
    private final PoolingSupport<NeighborArray> naturalScratch;
    private final PoolingSupport<NeighborArray> concurrentScratch;

    private final VectorSimilarityFunction similarityFunction;
    private final float neighborOverflow;
    private final float alpha;
    private final VectorEncoding vectorEncoding;
    private final PoolingSupport<GraphSearcher<?>> graphSearcher;

    @VisibleForTesting
    final OnHeapGraphIndex<T> graph;
    private final ConcurrentSkipListSet<Integer> insertionsInProgress = new ConcurrentSkipListSet<>();

    // We need two sources of vectors in order to perform diversity check comparisons without
    // colliding.  Usually it's obvious because you can see the different sources being used
    // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
    // and `vectorsCopy` later on when defining the ScoreFunction for search.
    private final PoolingSupport<RandomAccessVectorValues<T>> vectors;
    private final PoolingSupport<RandomAccessVectorValues<T>> vectorsCopy;
    private final NeighborSimilarity similarity;

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
    public GraphIndexBuilder(
            RandomAccessVectorValues<T> vectorValues,
            VectorEncoding vectorEncoding,
            VectorSimilarityFunction similarityFunction,
            int M,
            int beamWidth,
            float neighborOverflow,
            float alpha) {
        vectors = vectorValues.isValueShared() ? PoolingSupport.newThreadBased(vectorValues::copy) : PoolingSupport.newNoPooling(vectorValues);
        vectorsCopy = vectorValues.isValueShared() ? PoolingSupport.newThreadBased(vectorValues::copy) : PoolingSupport.newNoPooling(vectorValues);
        this.vectorEncoding = Objects.requireNonNull(vectorEncoding);
        this.similarityFunction = Objects.requireNonNull(similarityFunction);
        this.neighborOverflow = neighborOverflow;
        this.alpha = alpha;
        if (M <= 0) {
            throw new IllegalArgumentException("maxConn must be positive");
        }
        if (beamWidth <= 0) {
            throw new IllegalArgumentException("beamWidth must be positive");
        }
        this.beamWidth = beamWidth;

        similarity = node1 -> {
            try (var v = vectors.get(); var vc = vectorsCopy.get()) {
                T v1 = v.get().vectorValue(node1);
                return (NeighborSimilarity.ExactScoreFunction) node2 -> scoreBetween(v1, vc.get().vectorValue(node2));
            }
        };
        this.graph =
                new OnHeapGraphIndex<>(
                        M, (node, m) -> new ConcurrentNeighborSet(node, m, similarity, alpha));
        this.graphSearcher = PoolingSupport.newThreadBased(() -> new GraphSearcher.Builder<>(graph.getView()).withConcurrentUpdates().build());

        // in scratch we store candidates in reverse order: worse candidates are first
        this.naturalScratch = PoolingSupport.newThreadBased(() -> new NeighborArray(Math.max(beamWidth, M + 1)));
        this.concurrentScratch = PoolingSupport.newThreadBased(() -> new NeighborArray(Math.max(beamWidth, M + 1)));
    }

    public OnHeapGraphIndex<T> build() {
        int size;
        try (var v = vectors.get()) {
            size = v.get().size();
        }

        PhysicalCoreExecutor.instance.execute(() -> {
            IntStream.range(0, size).parallel().forEach(i -> {
                try (var v1 = vectors.get()) {
                    addGraphNode(i, v1.get());
                }
            });
        });

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

        // clean up overflowed neighbor lists
        IntStream.range(0, graph.getIdUpperBound()).parallel().forEach(i -> {
            var neighbors = graph.getNeighbors(i);
            if (neighbors != null) {
                neighbors.cleanup();
            }
        });

        // optimize entry node
        graph.updateEntryNode(approximateMedioid());
        graph.validateEntryNode(); // check again after updating
    }

    /**
     * @deprecated synonym for `cleanup`
     */
    public void complete() {
        cleanup();
    }

    public OnHeapGraphIndex<T> getGraph() {
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
     * @param node    the node ID to add
     * @param vectors the set of vectors
     * @return an estimate of the number of extra bytes used by the graph after adding the given node
     */
    public long addGraphNode(int node, RandomAccessVectorValues<T> vectors) {
        final T value = vectors.vectorValue(node);

    // do this before adding to in-progress, so a concurrent writer checking
    // the in-progress set doesn't have to worry about uninitialized neighbor sets
    var newNodeNeighbors = graph.addNode(node);

        insertionsInProgress.add(node);
        ConcurrentSkipListSet<Integer> inProgressBefore = insertionsInProgress.clone();
        try (var gs = graphSearcher.get();
             var vc = vectorsCopy.get();
             var naturalScratchPooled = naturalScratch.get();
             var concurrentScratchPooled = concurrentScratch.get()) {
            // find ANN of the new node by searching the graph
            int ep = graph.entry();
            NeighborSimilarity.ExactScoreFunction scoreFunction = i -> scoreBetween(vc.get().vectorValue(i), value);

            var bits = new ExcludingBits(node);
            // find best "natural" candidates with a beam search
            var result = gs.get().searchInternal(scoreFunction, null, beamWidth, ep, bits);

            // Update neighbors with these candidates.
            // The DiskANN paper calls for using the entire set of visited nodes along the search path as
            // potential candidates, but in practice we observe neighbor lists being completely filled using
            // just the topK results.  (Since the Robust Prune algorithm prioritizes closer neighbors,
            // this means that considering additional nodes from the search path, that are by definition
            // farther away than the ones in the topK, would not change the result.)
            // TODO if we made NeighborArray an interface we could wrap the NodeScore[] directly instead of copying
            var natural = toScratchCandidates(result.getNodes(), result.getNodes().length, naturalScratchPooled.get());
            var concurrent = getConcurrentCandidates(node, inProgressBefore, concurrentScratchPooled.get(), vectors, vc.get());
            updateNeighbors(newNodeNeighbors, natural, concurrent);
            graph.markComplete(node);
        } finally {
            insertionsInProgress.remove(node);
        }

        return graph.ramBytesUsedOneNode(0);
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
        var affectedLiveNodes = new HashSet<Integer>();
        var R = new Random();
        try (var v1 = vectors.get();
            var v2 = vectorsCopy.get())
        {
            for (var node : liveNodes) {
                assert !deletedNodes.get(node);

                ConcurrentNeighborSet neighbors = graph.getNeighbors(node);
                if (neighbors.removeDeletedNeighbors(deletedNodes)) {
                    affectedLiveNodes.add(node);
                }

                // add random connections if we've dropped below minimum
                int minConnections = 1 + graph.maxDegree() / 2;
                if (neighbors.size() < minConnections) {
                    // create a NeighborArray of random connections
                    NeighborArray randomConnections = new NeighborArray(graph.maxDegree() - neighbors.size());
                    // doing actual sampling-without-replacement is expensive so we'll loop a fixed number of times instead
                    for (int i = 0; i < 2 * graph.maxDegree(); i++) {
                        int randomNode = liveNodes[R.nextInt(liveNodes.length)];
                        if (randomNode != node && !randomConnections.contains(randomNode)) {
                            float score = scoreBetween(v1.get().vectorValue(node), v2.get().vectorValue(randomNode));
                            randomConnections.insertSorted(randomNode, score);
                        }
                        if (randomConnections.size == randomConnections.node.length) {
                            break;
                        }
                    }
                    neighbors.padWithRandom(randomConnections);
                }
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
        for (var node : affectedLiveNodes) {
            addNNDescentConnections(node);
        }

        // reset deleted collection
        deletedNodes.clear();

        return nRemoved * graph.ramBytesUsedOneNode(0);
    }

    /**
     * Search for the given node, then submit all nodes along the search path as candidates for
     * new neighbors.  Standard diversity pruning applies.
     */
    private void addNNDescentConnections(int node) {
        var notSelfBits = new Bits() {
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

        try (var gs = graphSearcher.get();
             var v1 = vectors.get();
             var v2 = vectorsCopy.get();
             var scratch = naturalScratch.get())
        {
            var value = v1.get().vectorValue(node);
            NeighborSimilarity.ExactScoreFunction scoreFunction = i -> scoreBetween(v2.get().vectorValue(i), value);
            var result = gs.get().searchInternal(scoreFunction, null, beamWidth, graph.entry(), notSelfBits);
            var candidates = getPathCandidates(result.getVisited(), node, scoreFunction, scratch.get());
            updateNeighbors(graph.getNeighbors(node), candidates, NeighborArray.EMPTY);
        }
    }

    private int approximateMedioid() {
        assert graph.size() > 0;

        if (vectorEncoding != VectorEncoding.FLOAT32) {
            // fill this in when/if we care about byte[] vectors
            return graph.entry();
        }

        try (var gs = graphSearcher.get();
             var vc = vectorsCopy.get())
        {
            // compute centroid
            var centroid = new float[vc.get().dimension()];
            for (var it = graph.getNodes(); it.hasNext(); ) {
                var node = it.nextInt();
                VectorUtil.addInPlace(centroid, (float[]) vc.get().vectorValue(node));
            }
            VectorUtil.divInPlace(centroid, graph.size());

            // search for the node closest to the centroid
            NeighborSimilarity.ExactScoreFunction scoreFunction = i -> scoreBetween(vc.get().vectorValue(i), (T) centroid);
            var result = gs.get().searchInternal(scoreFunction, null, beamWidth, graph.entry(), Bits.ALL);
            return result.getNodes()[0].node;
        }
    }

  private void updateNeighbors(ConcurrentNeighborSet neighbors, NeighborArray natural, NeighborArray concurrent) {
    neighbors.insertDiverse(natural, concurrent);
    neighbors.backlink(graph::getNeighbors, neighborOverflow);
  }

    /**
     * compute the scores for the nodes set in `visited` and return them in a NeighborArray
     */
    private NeighborArray getPathCandidates(BitSet visited, int node, NeighborSimilarity.ExactScoreFunction scoreFunction, NeighborArray scratch) {
        // doing a single sort is faster than repeatedly calling insertSorted
        SearchResult.NodeScore[] candidates = new SearchResult.NodeScore[visited.cardinality()];
        int j = 0;
        for (int i = visited.nextSetBit(0); i != NO_MORE_DOCS; i = visited.nextSetBit(i + 1)) {
            if (i != node) {
                candidates[j++] = new SearchResult.NodeScore(i, scoreFunction.similarityTo(i));
            }
        }
        Arrays.sort(candidates, 0, j, Comparator.comparingDouble(ns -> -ns.score));
        return toScratchCandidates(candidates, j, scratch);
    }

    private NeighborArray toScratchCandidates(SearchResult.NodeScore[] candidates, int count, NeighborArray scratch) {
        scratch.clear();
        for (int i = 0; i < count; i++) {
            var candidate = candidates[i];
            scratch.addInOrder(candidate.node, candidate.score);
        }
        return scratch;
    }

    private NeighborArray getConcurrentCandidates(int newNode,
                                                  Set<Integer> inProgress,
                                                  NeighborArray scratch,
                                                  RandomAccessVectorValues<T> values,
                                                  RandomAccessVectorValues<T> valuesCopy)
    {
        scratch.clear();
        for (var n : inProgress) {
            if (n != newNode) {
                scratch.insertSorted(
                        n,
                        scoreBetween(values.vectorValue(newNode), valuesCopy.vectorValue(n)));
            }
        }
        return scratch;
    }

    protected float scoreBetween(T v1, T v2) {
        return scoreBetween(vectorEncoding, similarityFunction, v1, v2);
    }

    static <T> float scoreBetween(
            VectorEncoding encoding, VectorSimilarityFunction similarityFunction, T v1, T v2) {
        switch (encoding) {
            case BYTE:
                return similarityFunction.compare((byte[]) v1, (byte[]) v2);
            case FLOAT32:
                return similarityFunction.compare((float[]) v1, (float[]) v2);
            default:
                throw new IllegalArgumentException();
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
            var ca = new NeighborArray(maxDegree);
            for (int j = 0; j < nNeighbors; j++) {
                int neighbor = in.readInt();
                ca.addInOrder(neighbor, similarity.score(node, neighbor));
            }
            graph.addNode(node, new ConcurrentNeighborSet(node, maxDegree, similarity, alpha, ca));
        }

        graph.updateEntryNode(entryNode);
    }
}
