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

import java.io.IOException;
import java.util.Arrays;

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
  int maxEdgesPerNode();

  @Override
  void close() throws IOException;

  interface View<T> extends AutoCloseable {
    /**
     * Iterator over the neighbors of a given node.  Only the most recently instantiated iterator
     * is guaranteed to be valid.
     */
    NodesIterator getNeighborsIterator(int node);

    int size();

    int entryNode();

    /**
     * Retrieve the vector associated with a given node.
     * <p>
     * This will only be called when a search is performed using approximate similarities.
     * In that situation, we will want to reorder the results by the exact similarity
     * at the end of the search.
     */
    T getVector(int node);

    // for compatibility with Cassandra's ExtendedHnswGraph.  Not sure if we still need it
    default int[] getSortedNodes() {
      int[] sortedNodes = new int[size()];
      Arrays.setAll(sortedNodes, i -> i);
      return sortedNodes;
    }

    //  for compatibility with Cassandra's ExtendedHnswGraph.  Not sure if we still want/need it
    default int getNeighborCount(int node) {
      return getNeighborsIterator(node).size();
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
