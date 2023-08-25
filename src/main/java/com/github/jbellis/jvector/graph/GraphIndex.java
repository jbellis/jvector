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

import java.io.IOException;

/**
 * TODO: add javadoc
 */
public interface GraphIndex {
  /** Returns the number of nodes in the graph */
  public int size();

  /**
   * Get all nodes on a given level as node 0th ordinals. The nodes are NOT guaranteed to be
   * presented in any particular order.
   *
   * @return an iterator over nodes where {@code nextInt} returns a next node on the level
   */
  public NodesIterator getNodes() throws IOException;

  /**
   * Add node on the given level with an empty set of neighbors.
   *
   * <p>Nodes can be inserted out of order, but it requires that the nodes preceded by the node
   * inserted out of order are eventually added.
   *
   * <p>Actually populating the neighbors, and establishing bidirectional links, is the
   * responsibility of the caller.
   *
   * <p>It is also the responsibility of the caller to ensure that each node is only added once.
   *
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int node);

  /**
   * Return a View with which to navigate the graph.  Views are not threadsafe.
   */
  public View getView();

  public interface View {
    /**
     * Move the pointer to exactly the given {@code level}'s {@code target}. After this method
     * returns, call {@link #nextNeighbor()} to return successive (ordered) connected node ordinals.
     *
     * @param target ordinal of a node in the graph, must be &ge; 0 and &lt;.
     */
    public void seek(int target) throws IOException;

    /**
     * Iterates over the neighbor list. It is illegal to call this method after it returns
     * NO_MORE_DOCS without calling {@link #seek(int)}, which resets the iterator.
     *
     * @return a node ordinal in the graph, or NO_MORE_DOCS if the iteration is complete.
     */
    public int nextNeighbor() throws IOException;

    public int size();

    public int entryNode();
  }
}
