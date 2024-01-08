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

import io.github.jbellis.jvector.util.AbstractLongHeap;
import io.github.jbellis.jvector.util.NumericUtils;

import java.util.stream.IntStream;

/**
 * NodeQueue uses a {@link io.github.jbellis.jvector.util.AbstractLongHeap} to store lists of nodes in a graph,
 * represented as a node id with an associated score packed together as a sortable long, which is sorted
 * primarily by score. The queue {@link #push(int, float)} operation provides either fixed-size
 * or unbounded operations, depending on the implementation subclasses, and either maxheap or minheap behavior.
 */
public class NodeQueue {

    public enum Order {
        MIN_HEAP {
            @Override
            long apply(long v) {
                return v;
            }
        },
        MAX_HEAP {
            @Override
            long apply(long v) {
                // This cannot be just `-v` since Long.MIN_VALUE doesn't have a positive counterpart. It
                // needs a function that returns MAX_VALUE for MIN_VALUE and vice-versa.
                return -1 - v;
            }
        };

        abstract long apply(long v);
    }

    private final AbstractLongHeap heap;
    private final Order order;

    // Whether the search stopped early because it reached the visited nodes limit
    private boolean incomplete;

    public NodeQueue(AbstractLongHeap heap, Order order) {
        this.heap = heap;
        this.order = order;
    }

    /**
     * @return the number of elements in the heap
     */
    public int size() {
        return heap.size();
    }

    /**
     * Adds a new graph node to the heap.  Will extend storage or replace the worst element
     * depending on the type of heap it is.
     *
     * @param newNode  the node id
     * @param newScore the relative similarity score to the node of the owner
     *
     * @return true if the new value was added.
     */
    public boolean push(int newNode, float newScore) {
        return heap.push(encode(newNode, newScore));
    }

    /**
     * Encodes the node ID and its similarity score as long.  If two scores are equals,
     * the smaller node ID wins.
     *
     * <p>The most significant 32 bits represent the float score, encoded as a sortable int.
     *
     * <p>The less significant 32 bits represent the node ID.
     *
     * <p>The bits representing the node ID are complemented to guarantee the win for the smaller node
     * Id.
     *
     * <p>The AND with 0xFFFFFFFFL (a long with first 32 bit as 1) is necessary to obtain a long that
     * has
     *
     * <p>The most significant 32 bits to 0
     *
     * <p>The less significant 32 bits represent the node ID.
     *
     * @param node  the node ID
     * @param score the node score
     * @return the encoded score, node ID
     */
    private long encode(int node, float score) {
        return order.apply(
                (((long) NumericUtils.floatToSortableInt(score)) << 32) | (0xFFFFFFFFL & ~node));
    }

    private float decodeScore(long heapValue) {
        return NumericUtils.sortableIntToFloat((int) (order.apply(heapValue) >> 32));
    }

    private int decodeNodeId(long heapValue) {
        return (int) ~(order.apply(heapValue));
    }

    /** Removes the top element and returns its node id. */
    public int pop() {
        return decodeNodeId(heap.pop());
    }

    /** Returns a copy of the internal nodes array. Not sorted by score! */
    public int[] nodesCopy() {
        int size = size();
        int[] nodes = new int[size];
        for (int i = 0; i < size; i++) {
            nodes[i] = decodeNodeId(heap.get(i + 1));
        }
        return nodes;
    }

    public SearchResult.NodeScore[] nodesCopy(NodeSimilarity.ExactScoreFunction sf, float rerankFloor) {
        return IntStream.range(0, size())
                .mapToObj(i -> heap.get(i + 1))
                .filter(m -> decodeScore(m) >= rerankFloor)
                .map(m -> new SearchResult.NodeScore(decodeNodeId(m), sf.similarityTo(decodeNodeId(m))))
                .toArray(SearchResult.NodeScore[]::new);
    }

    /** Returns the top element's node id. */
    public int topNode() {
        return decodeNodeId(heap.top());
    }

    /**
     * Returns the top element's node score. For the min heap this is the minimum score. For the max
     * heap this is the maximum score.
     */
    public float topScore() {
        return decodeScore(heap.top());
    }

    public void clear() {
        heap.clear();
        incomplete = false;
    }

    public boolean incomplete() {
        return incomplete;
    }

    public void markIncomplete() {
        this.incomplete = true;
    }

    @Override
    public String toString() {
        return "Nodes[" + heap.size() + "]";
    }
}
