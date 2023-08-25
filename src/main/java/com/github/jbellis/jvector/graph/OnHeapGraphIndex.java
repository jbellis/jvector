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

import com.github.jbellis.jvector.util.Accountable;
import com.github.jbellis.jvector.util.RamUsageEstimator;

import java.util.Map;
import java.util.PrimitiveIterator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.StampedLock;
import java.util.function.BiFunction;

import static com.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

/**
 * An {@link GraphIndex} that offers concurrent access; for typical graphs you will get significant
 * speedups in construction and searching as you add threads.
 *
 * <p>To search this graph, you should use a View obtained from {@link #getView()} to perform `seek`
 * and `nextNeighbor` operations.
 */
public final class OnHeapGraphIndex implements GraphIndex, Accountable {
  private final AtomicReference<Integer>
      entryPoint; // the current graph entry node on the top level. -1 if not set

  // Unlike OnHeapHnswGraph (OHHG), we use the same data structure for Level 0 and higher node
  // lists, a ConcurrentHashMap.  While the ArrayList used for L0 in OHHG is faster for
  // single-threaded workloads, it imposes an unacceptable contention burden for concurrent
  // graph building.
  private final Map<Integer, Map<Integer, ConcurrentNeighborSet>> graphLevels;
  private final CompletionTracker completions;

  // Neighbours' size on upper levels (nsize) and level 0 (nsize0)
  final int nsize;
  final int nsize0;
  private final BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory;

  OnHeapGraphIndex(
      int M, BiFunction<Integer, Integer, ConcurrentNeighborSet> neighborFactory) {
    this.neighborFactory = neighborFactory;
    this.entryPoint =
        new AtomicReference<>(-1); // Entry node should be negative until a node is added
    this.nsize = M;
    this.nsize0 = 2 * M;

    this.graphLevels = new ConcurrentHashMap<>();
    graphLevels.put(0, new ConcurrentHashMap<>());
    this.completions = new CompletionTracker(nsize0);
  }

  /**
   * Returns the neighbors connected to the given node.
   *
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public ConcurrentNeighborSet getNeighbors(int node) {
    return graphLevels.get(0).get(node);
  }

  @Override
  public int size() {
    Map<Integer, ConcurrentNeighborSet> levelZero = graphLevels.get(0);
    return levelZero == null ? 0 : levelZero.size(); // all nodes are located on the 0th level
  }

  public void addNode(int node) {
    graphLevels.get(0).put(node, neighborFactory.apply(node, connectionsOnLevel(0)));
  }

  /** must be called after addNode once neighbors are linked in all levels. */
  void markComplete(int node) {
    entryPoint.accumulateAndGet(
        node,
        (oldEntry, newEntry) -> {
          if (oldEntry >= 0) {
            return oldEntry;
          } else {
            return newEntry;
          }
        });
    completions.markComplete(node);
  }

  public void updateEntryNode(int node) {
    entryPoint.set(node);
  }

  private int connectionsOnLevel(int level) {
    return level == 0 ? nsize0 : nsize;
  }

  int entry() {
    return entryPoint.get();
  }

  @Override
  public NodesIterator getNodes() {
    // We avoid the temptation to optimize L0 by using ArrayNodesIterator.
    // This is because, while L0 will contain sequential ordinals once the graph is complete,
    // and internally Lucene only calls getNodesOnLevel at that point, this is a public
    // method so we cannot assume that that is the only time it will be called by third parties.
    return new NodesIterator.CollectionNodesIterator(graphLevels.get(0).keySet());
  }

  @Override
  public long ramBytesUsed() {
    // the main graph structure
    long total = concurrentHashMapRamUsed(graphLevels.size());
    Map<Integer, ConcurrentNeighborSet> level = graphLevels.get(0);
    int numNodesOnLevel = graphLevels.get(0).size();
    long chmSize = concurrentHashMapRamUsed(numNodesOnLevel);
    long neighborSize = neighborsRamUsed(connectionsOnLevel(0)) * numNodesOnLevel;

    total += chmSize + neighborSize;

    // logical clocks
    total += completions.ramBytesUsed();

    return total;
  }

  public long ramBytesUsedOneNode(int nodeLevel) {
    int entryCount = (int) (nodeLevel / CHM_LOAD_FACTOR);
    var graphBytesUsed =
        chmEntriesRamUsed(entryCount)
            + neighborsRamUsed(connectionsOnLevel(0))
            + nodeLevel * neighborsRamUsed(connectionsOnLevel(1));
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

  private static final float CHM_LOAD_FACTOR = 0.75f; // this is hardcoded inside ConcurrentHashMap

  /**
   * caller's responsibility to divide number of entries by load factor to get internal node count
   */
  private static long chmEntriesRamUsed(int internalEntryCount) {
    long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    long chmNodeBytes =
        REF_BYTES // node itself in Node[]
            + 3L * REF_BYTES
            + Integer.BYTES; // node internals

    return internalEntryCount * chmNodeBytes;
  }

  private static long concurrentHashMapRamUsed(int externalSize) {
    long REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
    long AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;
    long CORES = Runtime.getRuntime().availableProcessors();

    // CHM has a striped counter Cell implementation, we expect at most one per core
    long chmCounters = AH_BYTES + CORES * (REF_BYTES + Long.BYTES);

    int nodeCount = (int) (externalSize / CHM_LOAD_FACTOR);

    long chmSize =
        chmEntriesRamUsed(nodeCount) // nodes
            + nodeCount * REF_BYTES
            + AH_BYTES // nodes array
            + Long.BYTES
            + 3 * Integer.BYTES
            + 3 * REF_BYTES // extra internal fields
            + chmCounters
            + REF_BYTES; // the Map reference itself
    return chmSize;
  }

  @Override
  public String toString() {
    return "ConcurrentOnHeapHnswGraph(size=" + size() + ", entryPoint=" + entryPoint.get();
  }

  /**
   * Returns a view of the graph that is safe to use concurrently with updates performed on the
   * underlying graph.
   *
   * <p>Multiple Views may be searched concurrently.
   */
  public GraphIndex.View getView() {
    return new ConcurrentGraphIndexView();
  }

  void validateEntryNode() {
    if (size() == 0) {
      return;
    }
    var en = entryPoint.get();
    if (!(en >= 0 && graphLevels.get(0).containsKey(en))) {
      throw new IllegalStateException("Entry node was incompletely added! " + en);
    }
  }

  /**
   * A concurrent View of the graph that is safe to search concurrently with updates and with other
   * searches. The View provides a limited kind of snapshot isolation: only nodes completely added
   * to the graph at the time the View was created will be visible (but the connections between them
   * are allowed to change, so you could potentially get different top K results from the same query
   * if concurrent updates are in progress.)
   */
  private class ConcurrentGraphIndexView implements GraphIndex.View {
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
    private final int timestamp;
    private PrimitiveIterator.OfInt remainingNeighbors;

    public ConcurrentGraphIndexView() {
      this.timestamp = completions.clock();
    }

    @Override
    public void seek(int targetNode) {
      remainingNeighbors = getNeighbors(targetNode).nodeIterator();
    }

    @Override
    public int nextNeighbor() {
      while (remainingNeighbors.hasNext()) {
        int next = remainingNeighbors.nextInt();
        if (completions.completedAt(next) < timestamp) {
          return next;
        }
      }
      return NO_MORE_DOCS;
    }

    @Override
    public int size() {
      return OnHeapGraphIndex.this.size();
    }

    @Override
    public int entryNode() {
      return OnHeapGraphIndex.this.entryPoint.get();
    }

    @Override
    public String toString() {
      return "ConcurrentOnHeapHnswGraphView(size=" + size() + ", entryPoint=" + entryPoint.get();
    }
  }

  /** Class to provide snapshot isolation for nodes in the progress of being added. */
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

    /**
     * @param node ordinal
     */
    void markComplete(int node) {
      int completionClock = logicalClock.getAndIncrement();
      ensureCapacity(node);
      long stamp;
      do {
        stamp = sl.tryOptimisticRead();
        completionTimes.set(node, completionClock);
      } while (!sl.validate(stamp));
    }

    /**
     * @return the current logical timestamp; can be compared with completedAt values
     */
    int clock() {
      return logicalClock.get();
    }

    /**
     * @param node ordinal
     * @return the logical clock completion time of the node, or Integer.MAX_VALUE if the node has
     *     not yet been completed.
     */
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
      return REF_BYTES
          + Integer.BYTES // logicalClock
          + REF_BYTES
          + (long) Integer.BYTES * completionTimes.length();
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
          AtomicIntegerArray newArray = new AtomicIntegerArray(newSize);
          for (int i = 0; i < newSize; i++) {
            if (i < oldArray.length()) {
              newArray.set(i, oldArray.get(i));
            } else {
              newArray.set(i, Integer.MAX_VALUE);
            }
          }
          completionTimes = newArray;
        }
      } finally {
        sl.unlockWrite(stamp);
      }
    }
  }
}
