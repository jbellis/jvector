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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.ThreadSafeGrowableBitSet;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>To search this graph, you should use a View obtained from {@link #getView()} to perform `seek`
 * and `nextNeighbor` operations.
 */
public class OnHeapGraphIndex implements GraphIndex {
    static final int NO_ENTRY_POINT = -1;

    // the current graph entry node, NO_ENTRY_POINT if not set
    private final AtomicInteger entryPoint = new AtomicInteger(NO_ENTRY_POINT);

    final ConcurrentNeighborMap nodes;
    private final ThreadSafeGrowableBitSet deletedNodes = new ThreadSafeGrowableBitSet(0);
    private final AtomicInteger maxNodeId = new AtomicInteger(NO_ENTRY_POINT);

    // max neighbors/edges per node
    final int maxDegree;

    OnHeapGraphIndex(int M, int maxOverflowDegree, BuildScoreProvider scoreProvider, float alpha) {
        this.maxDegree = M;
        this.nodes = new ConcurrentNeighborMap(scoreProvider, maxDegree, maxOverflowDegree, alpha);
    }

    /**
     * Returns the neighbors connected to the given node, or null if the node does not exist.
     *
     * @param node the node whose neighbors are returned, represented as an ordinal.
     */
    ConcurrentNeighborMap.Neighbors getNeighbors(int node) {
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
     * @param nodeId the node to add, represented as an ordinal
     */
    public void addNode(int nodeId) {
        nodes.addNode(nodeId);
        maxNodeId.accumulateAndGet(nodeId, Math::max);
    }

    /**
     * Only for use by Builder loading a saved graph
     */
    void addNode(int nodeId, NodeArray nodes) {
        assert nodes != null;
        this.nodes.addNode(nodeId, nodes);
        maxNodeId.accumulateAndGet(nodeId, Math::max);
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
        return nodes.nodesIterator();
    }

    @Override
    public long ramBytesUsed() {
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        var REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        var AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long neighborSize = ramBytesUsedOneNode() * size();
        return OH_BYTES + REF_BYTES * 2L + AH_BYTES + neighborSize;
    }

    public long ramBytesUsedOneNode() {
        // we include the REF_BYTES for the CNS reference here to make it self-contained for addGraphNode()
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        return REF_BYTES + ConcurrentNeighborMap.Neighbors.ramBytesUsed(nodes.nodeArrayLength());
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
        return nodes.contains(nodeId);
    }

    public double getAverageDegree() {
        return IntStream.range(0, getIdUpperBound())
                .filter(this::containsNode)
                .mapToDouble(i -> getNeighbors(i).size())
                .average()
                .orElse(Double.NaN);
    }

    public void setScoreProvider(BuildScoreProvider bsp) {
        nodes.setScoreProvider(bsp);
    }

    public class ConcurrentGraphIndexView implements GraphIndex.View {
        public NodesIterator getNeighborsIterator(int node) {
            var neighbors = getNeighbors(node);
            assert neighbors != null : "Node " + node + " not found";
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
            nodes.forEach((nodeId, value) -> {
                try {
                    var neighbors = value.iterator();
                    out.writeInt(nodeId);

                    out.writeInt(neighbors.size());
                    for (int n = 0; n < neighbors.size(); n++) {
                        out.writeInt(neighbors.nextInt());
                    }
                    assert !neighbors.hasNext();
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
