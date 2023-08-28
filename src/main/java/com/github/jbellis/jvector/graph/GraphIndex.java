/*
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

package com.github.jbellis.jvector.graph;

import static com.github.jbellis.jvector.graph.NodesIterator.NO_MORE_NEIGHBORS;

/**
 * TODO: add javadoc
 */
public interface GraphIndex {
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
  View getView();

  interface View<T> {
    /**
     * Retrieve the vector associated with a given node.
     * <p/>
     * This will only be called when a search is performed using approximate similarities.
     * In that situation, we will want to reorder the results by the exact similarity
     * at the end of the search.
     */
    T getVector(int node);

    /**
     * Iterator over the neighbors of a given node.  Only the most recently instantiated iterator
     * is guaranteed to be valid.
     */
    NodesIterator getNeighborsIterator(int node);

    int size();

    int entryNode();
  }

  static String prettyPrint(GraphIndex graph) {
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
