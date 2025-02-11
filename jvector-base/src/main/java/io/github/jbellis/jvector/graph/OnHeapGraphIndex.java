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

import io.github.jbellis.jvector.graph.ConcurrentNeighborMap.Neighbors;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.SparseIntMap;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.StampedLock;
import java.util.stream.IntStream;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>The base layer (layer 0) contains all nodes, while higher layers are stored in sparse maps.
 * For searching, use a view obtained from {@link #getView()} which supports level–aware operations.
 */
public class OnHeapGraphIndex implements GraphIndex {
    // The current entry node for searches
    private final AtomicReference<NodeAtLevel> entryPoint;

    // Layers of the graph, with layer 0 as the bottom (dense) layer containing all nodes.
    final List<ConcurrentNeighborMap> layers = new ArrayList<>();

    private final CompletionTracker completions;
    private final ThreadSafeGrowableBitSet deletedNodes = new ThreadSafeGrowableBitSet(0);
    private final AtomicInteger maxNodeId = new AtomicInteger(-1);

    // Maximum number of neighbors (edges) per node in the base layer
    final int maxDegree;

    /**
     * Constructs a new OnHeapGraphIndex.
     *
     * @param M               maximum degree
     * @param maxOverflowDegree ratio by which construction is allowed to temporarily overflow the max degree
     * @param scoreProvider   a provider of build–time scores (may be used by node arrays)
     * @param alpha           diversity strictness parameter as described in DiskANN
     */
    OnHeapGraphIndex(int M, int maxOverflowDegree, BuildScoreProvider scoreProvider, float alpha) {
        this.maxDegree = M;
        entryPoint = new AtomicReference<>();
        this.completions = new CompletionTracker(1024);
        // Initialize the base layer (layer 0) with a dense map.
        this.layers.add(new ConcurrentNeighborMap(new DenseIntMap<>(1024), scoreProvider, maxDegree, maxOverflowDegree, alpha));
    }

    /**
     * Returns the neighbors for the given node at the specified level, or null if the node does not exist.
     *
     * @param level the layer
     * @param node  the node id
     * @return the Neighbors structure or null
     */
    Neighbors getNeighbors(int level, int node) {
        if (level >= layers.size()) {
            return null;
        }
        return layers.get(level).get(node);
    }

    @Override
    public int size() {
        // Base layer (layer 0) contains all nodes.
        return layers.get(0).size();
    }

    /**
     * Add the given node ordinal with an empty set of neighbors.
     *
     * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
     * inserted out of order are eventually added.
     *
     * <p>Actually populating the neighbors, and establishing bidirectional links, is the
     * responsibility of the caller.
     *
     * <p>It is also the responsibility of the caller to ensure that each node is only added once.
     *
     * @param nodeId the node to add, represented as an ordinal
     * @param maxLayer the maximum layer; entries will be initialized for the given node at all
     *                 layers L where 0 <= L <= maxLayer
     */
    public void addNode(NodeAtLevel nodeLevel) {
        // create extra layers if necessary
        for (int i = layers.size(); i <= nodeLevel.level; i++) {
            synchronized (layers) {
                if (i == layers.size()) { // doublecheck after locking
                    var denseMap = layers.get(0);
                    var map = new ConcurrentNeighborMap(new SparseIntMap<>(),
                                                        denseMap.scoreProvider,
                                                        maxDegree,
                                                        denseMap.maxOverflowDegree,
                                                        denseMap.alpha);
                    layers.add(map);
                }
            }
        }

        // add the node to each layer
        for (int i = 0; i <= nodeLevel.level; i++) {
            layers.get(i).addNode(nodeLevel.node);
        }
        maxNodeId.accumulateAndGet(nodeLevel.node, Math::max);
    }

    /**
     * Only for use by Builder loading a saved graph
     */
    void addNode(int level, int nodeId, NodeArray nodes) {
        assert nodes != null;
        this.layers.get(level).addNode(nodeId, nodes);
        maxNodeId.accumulateAndGet(nodeId, Math::max);
    }

    /**
     * Mark the given node deleted.  Does NOT remove the node from the graph.
     */
    public void markDeleted(int node) {
        deletedNodes.set(node);
    }

    /** must be called after addNode once neighbors are linked in all levels. */
    void markComplete(NodeAtLevel nodeLevel) {
        entryPoint.accumulateAndGet(
                nodeLevel,
                (oldEntry, newEntry) -> {
                    if (oldEntry == null || newEntry.level > oldEntry.level) {
                        return newEntry;
                    } else {
                        return oldEntry;
                    }
                });
        completions.markComplete(nodeLevel.node);
    }

    void updateEntryNode(NodeAtLevel newEntry) {
        entryPoint.set(newEntry);
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    NodeAtLevel entry() {
        return entryPoint.get();
    }

    @Override
    public NodesIterator getNodes(int layer) {
        return layers.get(layer).nodesIterator();
    }

    @Override
    public long ramBytesUsed() {
        var graphBytesUsed = IntStream.range(0, layers.size()).mapToLong(this::ramBytesUsedOneLayer).sum();
        return graphBytesUsed + completions.ramBytesUsed();
    }

    public long ramBytesUsedOneLayer(int layer) {
        // TODO Sparse and Dense layers are different
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        var REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        var AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long neighborSize = ramBytesUsedOneNode() * layers.get(layer).size();
        return OH_BYTES + REF_BYTES * 2L + AH_BYTES + neighborSize;
    }

    public long ramBytesUsedOneNode() {
        // we include the REF_BYTES for the CNS reference here to make it self-contained for addGraphNode()
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        // TODO different layers have different degrees
        return REF_BYTES + Neighbors.ramBytesUsed(layers.get(0).nodeArrayLength());
    }

    @Override
    public String toString() {
        return String.format("OnHeapGraphIndex(size=%d, entryPoint=%s)", size(), entryPoint.get());
    }

    @Override
    public void close() {
        // No resources to close.
    }

    /**
     * Returns a view of the graph that is safe to use concurrently with updates performed on the
     * underlying graph.
     *
     * <p>Multiple Views may be searched concurrently.
     */
    @Override
    public ConcurrentGraphIndexView getView() {
        return new ConcurrentGraphIndexView();
    }

    /**
     * Validates that the current entry node has been completely added.
     */
    void validateEntryNode() {
        if (size() == 0) {
            return;
        }
        NodeAtLevel entry = getView().entryNode();
        if (entry == null || getNeighbors(entry.level, entry.node) == null) {
            throw new IllegalStateException("Entry node was incompletely added! " + entry);
        }
    }

    public ThreadSafeGrowableBitSet getDeletedNodes() {
        return deletedNodes;
    }

    /**
     * Removes the given node from all layers.
     *
     * @param node the node id to remove
     * @return true if the node was present in any layer.
     */
    boolean removeNode(int node) {
        boolean found = false;
        for (var layer : layers) {
            if (layer.remove(node) != null) {
                found = true;
            }
        }
        deletedNodes.clear(node);
        return found;
    }

    @Override
    public int getIdUpperBound() {
        return maxNodeId.get() + 1;
    }

    @Override
    public boolean containsNode(int nodeId) {
        return layers.get(0).contains(nodeId);
    }

    /**
     * Returns the average degree computed over nodes in the base layer.
     *
     * @return the average degree or NaN if no nodes are present.
     */
    public double getAverageDegree(int level) {
        return IntStream.range(0, getIdUpperBound())
                .filter(this::containsNode)
                .mapToDouble(i -> getNeighbors(level, i).size())
                .average()
                .orElse(Double.NaN);
    }

    @Override
    public int getMaxLevel() {
        return layers.size() - 1;
    }

    public int getLayerSize(int level) {
        return layers.get(level).size();
    }

    /**
     * A concurrent View of the graph that is safe to search concurrently with updates and with other
     * searches. The View provides a limited kind of snapshot isolation: only nodes completely added
     * to the graph at the time the View was created will be visible (but the connections between them
     * are allowed to change, so you could potentially get different top K results from the same query
     * if concurrent updates are in progress.)
     */
    public class ConcurrentGraphIndexView implements View {
        // It is tempting, but incorrect, to try to provide "adequate" isolation by
        // (1) keeping a bitset of complete nodes and giving that to the searcher as nodes to
        // accept -- but we need to keep incomplete nodes out of the search path entirely,
        // not just out of the result set, or
        // (2) keeping a bitset of complete nodes and restricting the View to those nodes
        // -- but we needs to consider neighbor diversity separately for concurrent
        // inserts and completed nodes; this allows us to keep the former out of the latter,
        // but not the latter out of the former (when a node completes while we are working,
        // that was in-progress when we started.)
        // The only really foolproof solution is to implement snapshot isolation as
        // we have done here.
        private final int timestamp = completions.clock();

        @Override
        public NodesIterator getNeighborsIterator(int level, int node) {
            var it = getNeighbors(level, node).iterator();
            return new NodesIterator() {
                int nextNode = advance();

                private int advance() {
                    while (it.hasNext()) {
                        int n = it.nextInt();
                        if (completions.completedAt(n) < timestamp) {
                            return n;
                        }
                    }
                    return Integer.MIN_VALUE;
                }

                @Override
                public int size() {
                    throw new UnsupportedOperationException();
                }

                @Override
                public int nextInt() {
                    int current = nextNode;
                    if (current == Integer.MIN_VALUE) {
                        throw new IndexOutOfBoundsException();
                    }
                    nextNode = advance();
                    return current;
                }

                @Override
                public boolean hasNext() {
                    return nextNode != Integer.MIN_VALUE;
                }
            };
        }

        @Override
        public int size() {
            return OnHeapGraphIndex.this.size();
        }

        @Override
        public NodeAtLevel entryNode() {
            return entryPoint.get();
        }

        @Override
        public String toString() {
            NodeAtLevel entry = entryNode();
            return String.format("OnHeapGraphIndexView(size=%d, entryNode=%s)", size(), entry);
        }

        @Override
        public Bits liveNodes() {
            // this Bits will return true for node ids that no longer exist in the graph after being purged,
            // but we defined the method contract so that that is okay
            return deletedNodes.cardinality() == 0 ? Bits.ALL : Bits.inverseOf(deletedNodes);
        }

        @Override
        public int getIdUpperBound() {
            return OnHeapGraphIndex.this.getIdUpperBound();
        }

        @Override
        public void close() {
            // No resources to close.
        }
    }

    /**
     * Saves the graph to the given DataOutput for reloading into memory later
     */
    public void save(DataOutput out) {
        if (deletedNodes.cardinality() > 0) {
            throw new IllegalStateException("Cannot save a graph that has deleted nodes. Call cleanup() first");
        }

        try (var view = getView()) {
            // Write graph-level properties.
            out.writeInt(size());
            NodeAtLevel entry = view.entryNode();
            out.writeInt(entry.node);
            out.writeInt(maxDegree());

            // Save neighbors from the base layer.
            var baseLayer = layers.get(0);
            baseLayer.forEach((nodeId, neighbors) -> {
                try {
                    NodesIterator iterator = neighbors.iterator();
                    out.writeInt(nodeId);
                    out.writeInt(iterator.size());
                    for (int n = 0; n < iterator.size(); n++) {
                        out.writeInt(iterator.nextInt());
                    }
                    assert !iterator.hasNext();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * A helper class that tracks completion times for nodes.
     */
    static final class CompletionTracker implements Accountable {
        private final AtomicInteger logicalClock = new AtomicInteger();
        private volatile AtomicIntegerArray completionTimes;
        private final StampedLock sl = new StampedLock();

        public CompletionTracker(int initialSize) {
            completionTimes = new AtomicIntegerArray(initialSize);
            for (int i = 0; i < initialSize; i++) {
                completionTimes.set(i, Integer.MAX_VALUE);
            }
        }

        void markComplete(int node) {
            int completionClock = logicalClock.getAndIncrement();
            ensureCapacity(node);
            long stamp;
            do {
                stamp = sl.tryOptimisticRead();
                completionTimes.set(node, completionClock);
            } while (!sl.validate(stamp));
        }

        int clock() {
            return logicalClock.get();
        }

        public int completedAt(int node) {
            AtomicIntegerArray ct = completionTimes;
            if (node >= ct.length()) {
                return Integer.MAX_VALUE;
            }
            return ct.get(node);
        }

        @Override
        public long ramBytesUsed() {
            int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
            return REF_BYTES + Integer.BYTES + REF_BYTES +
                    (long) Integer.BYTES * completionTimes.length();
        }

        private void ensureCapacity(int node) {
            if (node < completionTimes.length()) {
                return;
            }
            long stamp = sl.writeLock();
            try {
                AtomicIntegerArray oldArray = completionTimes;
                if (node >= oldArray.length()) {
                    int newSize = (node + 1) * 2;
                    AtomicIntegerArray newArr = new AtomicIntegerArray(newSize);
                    for (int i = 0; i < newSize; i++) {
                        if (i < oldArray.length()) {
                            newArr.set(i, oldArray.get(i));
                        } else {
                            newArr.set(i, Integer.MAX_VALUE);
                        }
                    }
                    completionTimes = newArr;
                }
            } finally {
                sl.unlockWrite(stamp);
            }
        }
    }
}
