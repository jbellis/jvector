/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.stream.IntStream;

/**
 * Builder for Concurrent GraphIndex. See {@link GraphIndex} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 *
 * @param <T> the type of vector
 */
public class GraphIndexBuilder<T> {
  private final int beamWidth;
  private final PoolingSupport<NeighborArray> naturalScratch;
  private final PoolingSupport<NeighborArray> concurrentScratch;

  private final VectorSimilarityFunction similarityFunction;
  private final float neighborOverflow;
  private final VectorEncoding vectorEncoding;
  private final PoolingSupport<GraphSearcher<?>> graphSearcher;

  final OnHeapGraphIndex<T> graph;
  private final ConcurrentSkipListSet<Integer> insertionsInProgress =
          new ConcurrentSkipListSet<>();

  // We need two sources of vectors in order to perform diversity check comparisons without
  // colliding.  Usually it's obvious because you can see the different sources being used
  // in the same method.  The only tricky place is in addGraphNode, which uses `vectors` immediately,
  // and `vectorsCopy` later on when defining the ScoreFunction for search.
  private final PoolingSupport<RandomAccessVectorValues<T>> vectors;
  private final PoolingSupport<RandomAccessVectorValues<T>> vectorsCopy;

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param vectorValues the vectors whose relations are represented by the graph - must provide a
   *     different view over those vectors than the one used to add via addGraphNode.
   * @param M – the maximum number of connections a node can have
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   * @param neighborOverflow the ratio of extra neighbors to allow temporarily when inserting a
   *     node. larger values will build more efficiently, but use more memory.
   * @param alpha how aggressive pruning diverse neighbors should be.  Set alpha &gt; 1.0 to
   *        allow longer edges.  If alpha = 1.0 then the equivalent of the lowest level of
   *        an HNSW graph will be created, which is usually not what you want.
   */
  public GraphIndexBuilder(
          RandomAccessVectorValues<T> vectorValues,
          VectorEncoding vectorEncoding,
          VectorSimilarityFunction similarityFunction,
          int M,
          int beamWidth,
          float neighborOverflow,
          float alpha) {
    vectors = vectorValues.isValueShared() ? PoolingSupport.newThreadBased(vectorValues::copy) : PoolingSupport.newNoPooling(vectorValues);
    vectorsCopy = vectorValues.isValueShared() ? PoolingSupport.newThreadBased(vectorValues::copy) : PoolingSupport.newNoPooling(vectorValues);
    this.vectorEncoding = Objects.requireNonNull(vectorEncoding);
    this.similarityFunction = Objects.requireNonNull(similarityFunction);
    this.neighborOverflow = neighborOverflow;
    if (M <= 0) {
      throw new IllegalArgumentException("maxConn must be positive");
    }
    if (beamWidth <= 0) {
      throw new IllegalArgumentException("beamWidth must be positive");
    }
    this.beamWidth = beamWidth;

    NeighborSimilarity similarity = node1 -> {
      try(var v = vectors.get(); var vc = vectorsCopy.get()) {
        T v1 = v.get().vectorValue(node1);
        return (NeighborSimilarity.ExactScoreFunction) node2 -> scoreBetween(v1, vc.get().vectorValue(node2));
      }
    };
    this.graph =
            new OnHeapGraphIndex<>(
                    M, (node, m) -> new ConcurrentNeighborSet(node, m, similarity, alpha));
    this.graphSearcher = PoolingSupport.newThreadBased(() -> new GraphSearcher.Builder<>(graph.getView()).withConcurrentUpdates().build());

    // in scratch we store candidates in reverse order: worse candidates are first
    this.naturalScratch = PoolingSupport.newThreadBased(() -> new NeighborArray(Math.max(beamWidth, M + 1)));
    this.concurrentScratch = PoolingSupport.newThreadBased(() -> new NeighborArray(Math.max(beamWidth, M + 1)));
  }

  public OnHeapGraphIndex<T> build() {
    int size;
    try (var v = vectors.get()) {
      size = v.get().size();
    }

    PhysicalCoreExecutor.instance.execute(() -> {
      IntStream.range(0, size).parallel().forEach(i -> {
        try (var v1 = vectors.get()) {
          addGraphNode(i, v1.get());
        }
      });
    });

    complete();
    return graph;
  }

  public void complete() {
    graph.validateEntryNode(); // sanity check before we start
    PhysicalCoreExecutor.instance.execute(() -> IntStream.range(0, graph.size()).parallel().forEach(i -> graph.getNeighbors(i).cleanup()));
    graph.updateEntryNode(approximateMedioid());
    graph.validateEntryNode(); // check again after updating
  }


  public OnHeapGraphIndex<T> getGraph() {
    return graph;
  }

  /** Number of inserts in progress, across all threads. */
  public int insertsInProgress() {
    return insertionsInProgress.size();
  }

  /**
   * Inserts a node with the given vector value to the graph.
   *
   * <p>To allow correctness under concurrency, we track in-progress updates in a
   * ConcurrentSkipListSet. After adding ourselves, we take a snapshot of this set, and consider all
   * other in-progress updates as neighbor candidates.
   *
   * @param node the node ID to add
   * @param vectors the set of vectors
   * @return an estimate of the number of extra bytes used by the graph after adding the given node
   */
  public long addGraphNode(int node, RandomAccessVectorValues<T> vectors) {
    final T value = vectors.vectorValue(node);

    // do this before adding to in-progress, so a concurrent writer checking
    // the in-progress set doesn't have to worry about uninitialized neighbor sets
    graph.addNode(node);

    insertionsInProgress.add(node);
    ConcurrentSkipListSet<Integer> inProgressBefore = insertionsInProgress.clone();
    try (var gs = graphSearcher.get();
         var vc = vectorsCopy.get();
         var naturalScratchPooled = naturalScratch.get();
         var concurrentScratchPooled = concurrentScratch.get()){
      // find ANN of the new node by searching the graph
      int ep = graph.entry();
      NeighborSimilarity.ExactScoreFunction scoreFunction = i -> scoreBetween(vc.get().vectorValue(i), value);

      var bits = new ExcludingBits(node);
      // find best "natural" candidates with a beam search
      var candidates = gs.get().searchInternal(scoreFunction, null, beamWidth, ep, bits);

      // Update neighbors with these candidates.
      var natural = getNaturalCandidates(candidates.getNodes(), naturalScratchPooled.get());
      var concurrent = getConcurrentCandidates(node, inProgressBefore, concurrentScratchPooled.get(), vectors, vc.get());
      updateNeighbors(node, natural, concurrent);
      graph.markComplete(node);
    } finally {
      insertionsInProgress.remove(node);
    }

    return graph.ramBytesUsedOneNode(0);
  }

  private int approximateMedioid() {
    try (var v1 = vectors.get(); var v2 = vectorsCopy.get()) {
      GraphIndex.View<T> view = graph.getView();
      var startNode = view.entryNode();
      int newStartNode;

      // Check start node's neighbors for a better candidate, until we reach a local minimum.
      // This isn't a very good mediod approximation, but all we really need to accomplish is
      // not to be stuck with the worst possible candidate -- searching isn't super sensitive
      // to how good the mediod is, especially in higher dimensions
      while (true) {
        var startNeighbors = graph.getNeighbors(startNode).getCurrent();
        // Map each neighbor node to a pair of node and its average distance score.
        // (We use average instead of total, since nodes may have different numbers of neighbors.)
        newStartNode = IntStream.concat(IntStream.of(startNode), Arrays.stream(startNeighbors.node(), 0, startNeighbors.size))
                .mapToObj(node -> {
                  var nodeNeighbors = graph.getNeighbors(node).getCurrent();
                  double score = Arrays.stream(nodeNeighbors.node(), 0, nodeNeighbors.size)
                          .mapToDouble(i -> scoreBetween(v1.get().vectorValue(node), v2.get().vectorValue(i)))
                          .sum();
                  return new AbstractMap.SimpleEntry<>(node, score / v2.get().size());
                })
                // Find the entry with the minimum score
                .min(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                // Extract the node of the minimum entry
                .map(AbstractMap.SimpleEntry::getKey).get();
        if (startNode != newStartNode) {
          startNode = newStartNode;
        }
        else {
          return newStartNode;
        }
      }
    }
  }

  private void updateNeighbors(int node, NeighborArray natural, NeighborArray concurrent) {
    ConcurrentNeighborSet neighbors = graph.getNeighbors(node);
    neighbors.insertDiverse(natural, concurrent);
    neighbors.backlink(graph::getNeighbors, neighborOverflow);
  }

  private NeighborArray getNaturalCandidates(SearchResult.NodeScore[] candidates, NeighborArray scratch) {
    scratch.clear();
    for (SearchResult.NodeScore candidate : candidates) {
      scratch.addInOrder(candidate.node, candidate.score);
    }
    return scratch;
  }

  private NeighborArray getConcurrentCandidates(int newNode, Set<Integer> inProgress,
          NeighborArray scratch, RandomAccessVectorValues<T> values, RandomAccessVectorValues<T> valuesCopy) {
      scratch.clear();
      for (var n : inProgress) {
        if (n != newNode) {
          scratch.insertSorted(
                  n,
                  scoreBetween(values.vectorValue(newNode), valuesCopy.vectorValue(n)));
        }
      }
      return scratch;
  }

  protected float scoreBetween(T v1, T v2) {
    return scoreBetween(vectorEncoding, similarityFunction, v1, v2);
  }

  static <T> float scoreBetween(
          VectorEncoding encoding, VectorSimilarityFunction similarityFunction, T v1, T v2) {
    switch (encoding) {
      case BYTE:
        return similarityFunction.compare((byte[]) v1, (byte[]) v2);
      case FLOAT32:
        return similarityFunction.compare((float[]) v1, (float[]) v2);
      default:
        throw new IllegalArgumentException();
    }
  }

  private static class ExcludingBits implements Bits {
    private final int excluded;

    public ExcludingBits(int excluded) {
        this.excluded = excluded;
    }

    @Override
    public boolean get(int index) {
      return index != excluded;
    }

    @Override
    public int length() {
      throw new UnsupportedOperationException();
    }
  }
}
