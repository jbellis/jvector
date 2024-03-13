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

import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;

/**
 * Represents a graph-based vector index.  Nodes are represented as ints, and edges are
 * represented as adjacency lists.
 * <p>
 * Mostly this applies to any graph index, but a few methods (e.g. getVector()) are
 * specifically included to support the DiskANN-based design of OnDiskGraphIndex.
 * <p>
 * All methods are threadsafe.  Operations that require persistent state are wrapped
 * in a View that should be created per accessing thread.
 */
public interface GraphIndex extends AutoCloseable {
    /** Returns the number of nodes in the graph */
    int size();

    /**
     * Get all node ordinals included in the graph. The nodes are NOT guaranteed to be
     * presented in any particular order.
     *
     * @return an iterator over nodes where {@code nextInt} returns the next node.
     */
    NodesIterator getNodes();

    /**
     * Return a View with which to navigate the graph.  Views are not threadsafe -- that is,
     * only one search at a time should be run per View.
     * <p>
     * Additionally, the View represents a point of consistency in the graph, and in-use
     * Views prevent the removal of marked-deleted nodes from graphs that are being
     * concurrently modified.  Thus, it is good (and encouraged) to re-use Views for
     * on-disk, read-only graphs, but for in-memory graphs, it is better to create a new
     * View per search.
     */
    View getView();

    /**
     * @return the maximum number of edges per node
     */
    int maxDegree();

    /**
     * @return the maximum node id in the graph.  May be different from size() if nodes are
     * being added concurrently, or if nodes have been deleted (and cleaned up).
     */
    default int getIdUpperBound() {
        return size();
    }

    /**
     * @return true iff the graph contains the node with the given ordinal id
     */
    default boolean containsNode(int nodeId) {
        return nodeId >= 0 && nodeId < size();
    }

    @Override
    void close() throws IOException;

    interface View extends AutoCloseable {
        /**
         * Iterator over the neighbors of a given node.  Only the most recently instantiated iterator
         * is guaranteed to be valid.
         */
        NodesIterator getNeighborsIterator(int node);

        /**
         * @return the number of nodes in the graph
         */
        int size();

        /**
         * @return the node of the graph to start searches at
         */
        int entryNode();

        /**
         * Retrieve the vector associated with a given node.
         * <p>
         * This will only be called when a search is performed using approximate similarities.
         * In that situation, we will want to reorder the results by the exact similarity
         * at the end of the search.
         */
        VectorFloat<?> getVector(int node);

        /**
         * Retrieve the vector associated with a given node, and store it in the destination vector at the given offset.
         * @param node the node to retrieve
         * @param destinationVector the vector to store the result in
         * @param offset the offset in the destination vector to store the result
         */
        void getVectorInto(int node, VectorFloat<?> destinationVector, int offset);

        /**
         * Return a Bits instance indicating which nodes are live.  The result is undefined for
         * ordinals that do not correspond to nodes in the graph.
         */
        Bits liveNodes();

        /**
         * @return the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
         */
        default int getIdUpperBound() {
            return size();
        }
    }

    static String prettyPrint(GraphIndex graph) {
        StringBuilder sb = new StringBuilder();
        sb.append(graph);
        sb.append("\n");

        try (var view = graph.getView()) {
            NodesIterator it = graph.getNodes();
            while (it.hasNext()) {
                int node = it.nextInt();
                sb.append("  ").append(node).append(" -> ");
                for (var neighbors = view.getNeighborsIterator(node); neighbors.hasNext(); ) {
                    sb.append(" ").append(neighbors.nextInt());
                }
                sb.append("\n");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return sb.toString();
    }
}
