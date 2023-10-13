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
public interface GraphIndex<T> extends AutoCloseable {
  /** Returns the number of nodes in the graph */
  int size();

  /**
   * Get all nodes on a given level as node 0th ordinals. The nodes are NOT guaranteed to be
   * presented in any particular order.
   *
   * @return an iterator over nodes where {@code nextInt} returns a next node on the level
   */
  NodesIterator getNodes();

  /**
   * Return a View with which to navigate the graph.  Views are not threadsafe.
   */
  View<T> getView();

  /**
   * @return the maximum number of edges per node
   */
  int maxDegree();

  /**
   * @return the maximum node id in the graph.  May be different from size() if nodes are
   * being added concurrently, or if nodes have been deleted (and cleaned up).
   */
  default int getMaxNodeId() {
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

  interface View<T> extends AutoCloseable {
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
    T getVector(int node);

    /**
     * Return a Bits instance indicating which nodes are live.  The result is undefined for
     * ordinals that do not correspond to nodes in the graph.
     */
    Bits liveNodes();

    /**
     * @return the largest ordinal id in the graph.  May be different from size() if nodes have been deleted.
     */
    default int getMaxNodeId() {
      return size();
    }
  }

  static <T> String prettyPrint(GraphIndex<T> graph) {
    StringBuilder sb = new StringBuilder();
    sb.append(graph);
    sb.append("\n");

    var view = graph.getView();
    NodesIterator it = graph.getNodes();
    while (it.hasNext()) {
      int node = it.nextInt();
      sb.append("  ").append(node).append(" -> ");
      for (var neighbors = view.getNeighborsIterator(node); neighbors.hasNext(); ) {
        sb.append(" ").append(neighbors.nextInt());
      }
      sb.append("\n");
    }

    return sb.toString();
  }
}
