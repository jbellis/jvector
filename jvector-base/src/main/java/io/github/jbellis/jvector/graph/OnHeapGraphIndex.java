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

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>To search this graph, you should use a View obtained from {@link #getView()} to perform `seek`
 * and `nextNeighbor` operations.
 */
public class OnHeapGraphIndex implements GraphIndex, Accountable {
    static final int NO_ENTRY_POINT = -1;

    // the current graph entry node, NO_ENTRY_POINT if not set
    private final AtomicInteger entryPoint = new AtomicInteger(NO_ENTRY_POINT);

    private final DenseIntMap<ConcurrentNeighborSet> nodes;
    private final ThreadSafeGrowableBitSet deletedNodes = new ThreadSafeGrowableBitSet(0);
    private final AtomicInteger maxNodeId = new AtomicInteger(NO_ENTRY_POINT);

    // max neighbors/edges per node
    final int maxDegree;
    private final BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory;

    OnHeapGraphIndex(int M, BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory) {
        this.neighborFactory = neighborFactory;
        this.maxDegree = M;
        this.nodes = new DenseIntMap<>(1024);
    }

    /**
     * Returns the neighbors connected to the given node, or null if the node does not exist.
     *
     * @param node the node whose neighbors are returned, represented as an ordinal.
     */
    ConcurrentNeighborSet getNeighbors(int node) {
        return nodes.get(node);
    }


    @Override
    public int size() {
        return nodes.size();
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
     * @param node the node to add, represented as an ordinal
     * @return the neighbor set for this node
     */
    public ConcurrentNeighborSet addNode(int node) {
        var newNeighborSet = neighborFactory.apply(node, maxDegree());
        addNode(node, newNeighborSet);
        return newNeighborSet;
    }

    /**
     * Only for use by Builder loading a saved graph
     */
    void addNode(int node, ConcurrentNeighborSet neighbors) {
        assert neighbors != null;
        nodes.put(node, neighbors);
        maxNodeId.accumulateAndGet(node, Math::max);
    }

    /**
     * Mark the given node deleted.  Does NOT remove the node from the graph.
     */
    public void markDeleted(int node) {
        deletedNodes.set(node);
    }

    /** must be called after addNode once neighbors are linked */
    void maybeSetInitialEntryNode(int node) {
        entryPoint.accumulateAndGet(node,
                                    (oldEntry, newEntry) -> {
                                        if (oldEntry >= 0) {
                                            return oldEntry;
                                        } else {
                                            return newEntry;
                                        }
                                    });
    }

    void updateEntryNode(int node) {
        entryPoint.set(node);
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    int entry() {
        return entryPoint.get();
    }

    @Override
    public NodesIterator getNodes() {
        return nodes.getNodesIterator();
    }

    @Override
    public long ramBytesUsed() {
        // the main graph structure
        long total = (long) size() * RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        long neighborSize = neighborsRamUsed(maxDegree()) * size();
        return total + neighborSize + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
    }

    public long ramBytesUsedOneNode() {
        var graphBytesUsed =
                neighborsRamUsed(maxDegree());
        var clockBytesUsed = Integer.BYTES;
        return graphBytesUsed + clockBytesUsed;
    }

    private static long neighborsRamUsed(int count) {
        long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        long AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
        long neighborSetBytes =
                REF_BYTES // atomicreference
                        + Integer.BYTES
                        + Integer.BYTES
                        + REF_BYTES // NeighborArray
                        + AH_BYTES * 2 // NeighborArray internals
                        + REF_BYTES * 2
                        + Integer.BYTES
                        + 1;
        return neighborSetBytes + (long) count * (Integer.BYTES + Float.BYTES);
    }


    @Override
    public String toString() {
        return String.format("OnHeapGraphIndex(size=%d, entryPoint=%d)", size(), entryPoint.get());
    }

    @Override
    public void close() {
        // no-op
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

    void validateEntryNode() {
        if (size() == 0) {
            return;
        }
        var en = entryPoint.get();
        if (!(en >= 0 && getNeighbors(en) != null)) {
            throw new IllegalStateException("Entry node was incompletely added! " + en);
        }
    }

    public ThreadSafeGrowableBitSet getDeletedNodes() {
        return deletedNodes;
    }

    /**
     * @return true iff the node was present.
     */
    boolean removeNode(int node) {
        try {
            return nodes.remove(node) != null;
        } finally {
            deletedNodes.clear(node);
        }
    }

    @Override
    public int getIdUpperBound() {
        return maxNodeId.get() + 1;
    }

    public boolean containsNode(int nodeId) {
        return nodes.containsKey(nodeId);
    }

    public double getAverageShortEdges() {
        return IntStream.range(0, getIdUpperBound())
                .filter(this::containsNode)
                .mapToDouble(i -> getNeighbors(i).getShortEdges())
                .average()
                .orElse(Double.NaN);
    }

    public double getAverageDegree() {
        return IntStream.range(0, getIdUpperBound())
                .filter(this::containsNode)
                .mapToDouble(i -> getNeighbors(i).size())
                .average()
                .orElse(Double.NaN);
    }

    // These allow GraphIndexBuilder to tell when it is safe to remove nodes from the graph.
    AtomicLong viewStamp = new AtomicLong(0);
    Set<Long> activeViews = ConcurrentHashMap.newKeySet();
    @SuppressWarnings("BusyWait")
    public void waitForViewsBefore(long stamp) {
        while (activeViews.stream().anyMatch(v -> v < stamp)) {
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    class ConcurrentGraphIndexView implements GraphIndex.View {
        final long stamp;

        public ConcurrentGraphIndexView() {
            stamp = viewStamp.incrementAndGet();
            activeViews.add(stamp);
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            throw new UnsupportedOperationException("All searches done with OnHeapGraphIndex should be exact");
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            throw new UnsupportedOperationException("All searches done with OnHeapGraphIndex should be exact");
        }

        public NodesIterator getNeighborsIterator(int node) {
            var neighbors = getNeighbors(node);
            assert neighbors != null : "Node " + node + " not found @" + stamp;
            return neighbors.iterator();
        }

        @Override
        public int size() {
            return OnHeapGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return entryPoint.get();
        }

        @Override
        public String toString() {
            return "OnHeapGraphIndexView(size=" + size() + ", entryPoint=" + entryPoint.get();
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
            activeViews.remove(stamp);
        }
    }

    public void save(DataOutput out) {
        if (deletedNodes.cardinality() > 0) {
            throw new IllegalStateException("Cannot save a graph that has deleted nodes.  Call cleanup() first");
        }

        // graph-level properties
        try (var view = getView()) {
            out.writeInt(size());
            out.writeInt(view.entryNode());
            out.writeInt(maxDegree());

            // neighbors
            for (var entry : nodes.entrySet()) {
                var i = (int) (long) entry.getKey();
                var neighbors = entry.getValue().iterator();
                out.writeInt(i);

                out.writeInt(neighbors.size());
                for (int n = 0; n < neighbors.size(); n++) {
                    out.writeInt(neighbors.nextInt());
                }
                assert !neighbors.hasNext();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
