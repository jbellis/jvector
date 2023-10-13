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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.IntStream;

public class OnDiskGraphIndex<T> implements GraphIndex<T>, AutoCloseable, Accountable
{
    private final ReaderSupplier readerSupplier;
    private final long neighborsOffset;
    private final int size;
    private final int entryNode;
    private final int maxDegree;
    private final int dimension;

    public OnDiskGraphIndex(ReaderSupplier readerSupplier, long offset)
    {
        this.readerSupplier = readerSupplier;
        this.neighborsOffset = offset + 4 * Integer.BYTES;
        try (var reader = readerSupplier.get()) {
            reader.seek(offset);
            size = reader.readInt();
            dimension = reader.readInt();
            entryNode = reader.readInt();
            maxDegree = reader.readInt();
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i < j in `graph` then map[i] < map[j] in the returned map.
     */
    public static Map<Integer, Integer> getSequentialRenumbering(OnHeapGraphIndex<float[]> graph) {
        var view = graph.getView();
        Map<Integer, Integer> oldToNewMap = new HashMap<>();
        int nextOrdinal = 0;
        for (int i = 0; i <= view.getMaxNodeId(); i++) {
            if (graph.containsNode(i)) {
                oldToNewMap.put(i, nextOrdinal++);
            }
        }
        return oldToNewMap;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    /** return a Graph that can be safely queried concurrently */
    public OnDiskGraphIndex<T>.OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    // TODO: This is fake generic until the reading functionality uses T
    public class OnDiskView implements GraphIndex.View<T>, AutoCloseable
    {
        private final RandomAccessReader reader;
        private final int[] neighbors;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
            this.neighbors = new int[maxDegree];
        }

        public T getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                float[] vector = new float[dimension];
                reader.seek(offset);
                reader.readFully(vector);
                return (T) vector;
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public NodesIterator getNeighborsIterator(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES) +
                        (node * (long) Integer.BYTES * (maxDegree + 1)));
                int neighborCount = reader.readInt();
                assert neighborCount <= maxDegree : String.format("neighborCount %d > M %d", neighborCount, maxDegree);
                reader.read(neighbors, 0, neighborCount);
                return new NodesIterator.ArrayNodesIterator(neighbors, neighborCount);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public int size() {
            return OnDiskGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return OnDiskGraphIndex.this.entryNode;
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }

    @Override
    public NodesIterator getNodes()
    {
        return NodesIterator.fromPrimitiveIterator(IntStream.range(0, size).iterator(), size);
    }

    @Override
    public long ramBytesUsed() {
        return Long.BYTES + 4 * Integer.BYTES;
    }

    public void close() throws IOException {
        readerSupplier.close();
    }

    /**
     * @param graph the graph to write
     * @param vectors the vectors associated with each node
     * @param ordinalMapper A function that maps from the graph's ordinals to the ordinals in the output.
     *                      For simple use cases this can be the identity function.  To deal with deleted nodes,
     *                      or to renumber the nodes to match rows or documents elsewhere, you may provide
     *                      something custom.  The mapper must map from the graph's ordinals
     *                      [0..getMaxNodeId()], to the output's ordinals [0..size()), with no gaps and
     *                      no duplicates.
     * @param out the output to write to
     */
    public static <T> void write(GraphIndex<T> graph,
                                 RandomAccessVectorValues<T> vectors,
                                 Function<Integer, Integer> ordinalMapper,
                                 DataOutput out)
            throws IOException
    {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex<T>) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }

        var view = graph.getView();

        // graph-level properties
        out.writeInt(graph.size());
        out.writeInt(vectors.dimension());
        out.writeInt(view.entryNode());
        out.writeInt(graph.maxDegree());

        // for each graph node, write the associated vector and its neighbors
        var newOrdinals = new HashSet<Integer>();
        for (int originalOrdinal = 0; originalOrdinal <= graph.getMaxNodeId(); originalOrdinal++) {
            if (!graph.containsNode(originalOrdinal)) {
                continue;
            }

            int newOrdinal = ordinalMapper.apply(originalOrdinal);
            newOrdinals.add(newOrdinal);
            out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            Io.writeFloats(out, (float[]) vectors.vectorValue(originalOrdinal));

            var neighbors = view.getNeighborsIterator(originalOrdinal);
            out.writeInt(neighbors.size());
            int n = 0;
            for ( ; n < neighbors.size(); n++) {
                out.writeInt(ordinalMapper.apply(neighbors.nextInt()));
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.maxDegree(); n++) {
                out.writeInt(-1);
            }
        }

        // verify that the provided mapper was well-behaved
        if (newOrdinals.size() > graph.size()) {
            throw new IllegalArgumentException("graph modified during write");
        }
        if (newOrdinals.size() < graph.size()) {
            throw new IllegalArgumentException("ordinalMapper resulted in duplicate entries");
        }
        if (graph.size() > 0 && newOrdinals.stream().mapToInt(i -> i).max().getAsInt() != graph.size() - 1)  {
            throw new IllegalArgumentException("ordinalMapper produced out-of-range entries");
        }
    }
}
