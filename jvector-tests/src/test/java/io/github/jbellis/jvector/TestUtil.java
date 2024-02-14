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

package io.github.jbellis.jvector;

import io.github.jbellis.jvector.disk.OnDiskADCGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.carrotsearch.randomizedtesting.RandomizedTest.getRandom;
import static org.junit.Assert.assertEquals;

public class TestUtil {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /** min .. max inclusive on both ends, to match Lucene's */
    public static int nextInt(Random random, int min, int max) {
        return min + random.nextInt(1 + max - min);
    }

    public static DataOutputStream openFileForWriting(Path outputPath) throws IOException {
        return new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(outputPath)));
    }

    /**
     * Deletes target Path and its contents (if applicable). Never throws an exception. Only suitable for tests.
     * Visits all levels of file tree and does not follow symlinks.
     * Exceptions will terminate the walk and print errors on stderr.
     * @param targetPath Path to delete
     */
    public static void deleteQuietly(Path targetPath) {
        try {
            Files.walkFileTree(targetPath, new FileVisitor<>() {
                @Override
                public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    Files.delete(file);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
                    System.err.println("deleteQuietly encountered an Exception when visiting file: " + exc.toString());
                    return FileVisitResult.TERMINATE;

                }

                @Override
                public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                    if (exc != null) {
                        System.err.println("deleteQuietly encountered an Exception after visiting directory: " + exc);
                        return FileVisitResult.TERMINATE;
                    }
                    Files.delete(dir);
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            System.err.println("deleteQuietly encountered an Exception: " + e);
        }
    }

    public static VectorFloat<?> randomVector(Random random, int dim) {
        var vec = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, random.nextFloat());
            if (random.nextBoolean()) {
                vec.set(i, -vec.get(i));
            }
        }
        VectorUtil.l2normalize(vec);
        return vec;
    }

    public static ByteSequence<?> randomVector8(VectorTypeSupport vts, Random random, int dim) {
        VectorFloat<?> fvec = randomVector(random, dim);
        ByteSequence<?> bvec = vts.createByteSequence(dim);
        for (int i = 0; i < dim; i++) {
            bvec.set(i, (byte) (fvec.get(i) * 127));
        }
        return bvec;
    }

    public static List<VectorFloat<?>> createRandomVectors(int count, int dimension) {
        return IntStream.range(0, count).mapToObj(i -> TestUtil.randomVector(getRandom(), dimension)).collect(Collectors.toList());
    }

    public static <T> void writeGraph(GraphIndex graph, RandomAccessVectorValues vectors, Path outputPath) throws IOException {
        try (var out = openFileForWriting(outputPath))
        {
            OnDiskGraphIndex.write(graph, vectors, out);
            out.flush();
        }
    }

    public static <T> void writeFusedGraph(GraphIndex graph, RandomAccessVectorValues vectors, PQVectors pq, Path outputPath) throws IOException {
        try (var out = openFileForWriting(outputPath))
        {
            OnDiskADCGraphIndex.write(graph, vectors, pq, out);
            out.flush();
        }
    }

    public static Set<Integer> getNeighborNodes(GraphIndex.View g, int node) {
      Set<Integer> neighbors = new HashSet<>();
      for (var it = g.getNeighborsIterator(node); it.hasNext(); ) {
        int n = it.nextInt();
        neighbors.add(n);
      }
      return neighbors;
    }

    static List<Integer> sortedNodes(GraphIndex h) {
          var graphNodes = h.getNodes();
          List<Integer> nodes = new ArrayList<>();
          while (graphNodes.hasNext()) {
              nodes.add(graphNodes.next());
          }
          Collections.sort(nodes);
          return nodes;
      }

    public static void assertGraphEquals(GraphIndex g, GraphIndex h) {
          // construct these up front since they call seek which will mess up our test loop
          String prettyG = GraphIndex.prettyPrint(g);
          String prettyH = GraphIndex.prettyPrint(h);
          assertEquals(String.format("the number of nodes in the graphs are different:%n%s%n%s",
                                     prettyG,
                                     prettyH),
                       g.size(),
                       h.size());

          // assert equal nodes in each graph
          List<Integer> hNodes = sortedNodes(h);
          List<Integer> gNodes = sortedNodes(g);
          assertEquals(String.format("nodes in the graphs are different:%n%s%n%s",
                                     prettyG,
                                     prettyH),
                       gNodes,
                       hNodes);

          // assert equal nodes' neighbours in each graph
          NodesIterator gNodesIterator = g.getNodes();
          var gv = g.getView();
          var hv = h.getView();
          while (gNodesIterator.hasNext()) {
              int node = gNodesIterator.nextInt();
              assertEqualsLazy(() -> String.format("arcs differ for node %d%n%s%n%s",
                                                   node,
                                                   prettyG,
                                                   prettyH),
                               getNeighborNodes(gv, node),
                               getNeighborNodes(hv, node));
          }
      }

    /**
     * For when building the failure message is expensive
     */
    public static void assertEqualsLazy(Supplier<String> f, Set<Integer> s1, Set<Integer> s2) {
        if (!s1.equals(s2)) {
            throw new AssertionError(f.get());
        }
    }

    public static <T> OnHeapGraphIndex buildSequentially(GraphIndexBuilder builder, RandomAccessVectorValues vectors) {
        for (var i = 0; i < vectors.size(); i++) {
            builder.addGraphNode(i, vectors);
        }
        builder.cleanup();
        return builder.getGraph();
    }

    public static class FullyConnectedGraphIndex implements GraphIndex {
        private final int entryNode;
        private final int size;

        public FullyConnectedGraphIndex(int entryNode, int size) {
            this.entryNode = entryNode;
            this.size = size;
        }

        @Override
        public int size()
        {
            return size;
        }

        @Override
        public NodesIterator getNodes() {
            return new NodesIterator.ArrayNodesIterator(IntStream.range(0, size).toArray(),  size);
        }

        @Override
        public View getView() {
            return new FullyConnectedGraphIndexView();
        }

        @Override
        public int maxDegree() {
            return size - 1;
        }

        @Override
        public void close() { }

        private class FullyConnectedGraphIndexView implements View {
            @Override
            public NodesIterator getNeighborsIterator(int node) {
                return new NodesIterator.ArrayNodesIterator(IntStream.range(0, size).filter(i -> i != node).toArray() , size - 1);
            }

            @Override
            public int size() {
                return size;
            }

            @Override
            public int entryNode() {
                return entryNode;
            }

            @Override
            public VectorFloat<?> getVector(int node) {
                throw new UnsupportedOperationException("No vectors associated with FullyConnectedGraphIndex");
            }

            @Override
            public Bits liveNodes() {
                return Bits.ALL;
            }

            @Override
            public void close() { }
        }
    }

    public static class RandomlyConnectedGraphIndex implements GraphIndex {
        private final int size;
        private final Map<Integer, int[]> nodes;
        private final int entryNode;

        public RandomlyConnectedGraphIndex(int size, int M, Random random) {
            this.size = size;
            this.nodes = new ConcurrentHashMap<>();

            var maxNeighbors = Math.min(M, size - 1);
            var nodeIds = IntStream.range(0, size).boxed().collect(Collectors.toCollection(ArrayList::new));
            Collections.shuffle(nodeIds, random);

            for (int i = 0; i < size; i++) {
                Set<Integer> neighborSet = new HashSet<>();
                while (neighborSet.size() < maxNeighbors) {
                    var neighborIdx = random.nextInt(size);
                    if (neighborIdx != i) {
                        neighborSet.add(nodeIds.get(neighborIdx));
                    }
                    nodes.put(nodeIds.get(i), neighborSet.stream().mapToInt(Integer::intValue).toArray());
                }
            }

            this.entryNode = random.nextInt(size);
        }

        @Override
        public int size()
        {
            return size;
        }

        @Override
        public NodesIterator getNodes() {
            return new NodesIterator.ArrayNodesIterator(IntStream.range(0, size).toArray(),  size);
        }

        @Override
        public View getView() {
            return new RandomlyConnectedGraphIndexView();
        }

        @Override
        public int maxDegree() {
            return nodes.get(0).length;
        }

        @Override
        public void close() { }

        private class RandomlyConnectedGraphIndexView implements View {
            @Override
            public NodesIterator getNeighborsIterator(int node) {
                return new NodesIterator.ArrayNodesIterator(nodes.get(node));
            }

            @Override
            public int size() {
                return size;
            }

            @Override
            public int entryNode() {
                return entryNode;
            }

            @Override
            public VectorFloat<?> getVector(int node) {
                throw new UnsupportedOperationException("No vectors associated with RandomlyConnectedGraphIndex");
            }

            @Override
            public Bits liveNodes() {
                return Bits.ALL;
            }

            @Override
            public void close() { }
        }
    }
}
