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
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2IntHashMap;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class OnDiskGraphIndex implements GraphIndex, AutoCloseable, Accountable
{
    protected static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    protected final ReaderSupplier readerSupplier;
    protected final long neighborsOffset;
    protected final int size;
    protected final int entryNode;
    protected final int maxDegree;
    protected final int dimension;

    protected OnDiskGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighborsOffset)
    {
        this.readerSupplier = readerSupplier;
        this.neighborsOffset = neighborsOffset;
        this.size = info.size;
        this.entryNode = info.entryNode;
        this.maxDegree = info.maxDegree;
        this.dimension = info.dimension;
    }

    public static OnDiskGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            return new OnDiskGraphIndex(readerSupplier, info, offset + 4 * Integer.BYTES);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /**
     * Header information common to all on-disk graphs
     */
    protected static class CommonHeader {
        public static final long SIZE = 4 * Integer.BYTES;

        public final int size;
        public final int dimension;
        public final int entryNode;
        public final int maxDegree;

        public CommonHeader(int size, int dimension, int entryNode, int maxDegree) {
            this.size = size;
            this.dimension = dimension;
            this.entryNode = entryNode;
            this.maxDegree = maxDegree;
        }

        public static CommonHeader load(RandomAccessReader reader, long offset) throws IOException {
            reader.seek(offset);
            int size = reader.readInt();
            int dimension = reader.readInt();
            int entryNode = reader.readInt();
            int maxDegree = reader.readInt();
            return new CommonHeader(size, dimension, entryNode, maxDegree);
        }
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.
     */
    public static Map<Integer, Integer> getSequentialRenumbering(GraphIndex graph) {
        try (var view = graph.getView()) {
            Int2IntHashMap oldToNewMap = new Int2IntHashMap(-1);
            int nextOrdinal = 0;
            for (int i = 0; i < view.getIdUpperBound(); i++) {
                if (graph.containsNode(i)) {
                    oldToNewMap.put(i, nextOrdinal++);
                }
            }
            return oldToNewMap;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
    public ViewWithVectors getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    public class OnDiskView implements GraphIndex.ViewWithVectors, AutoCloseable
    {
        private final RandomAccessReader reader;
        private final int[] neighbors;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
            this.neighbors = new int[maxDegree];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            return this;
        }

        public VectorFloat<?> getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            try {
                long diskOffset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(diskOffset);
                vectorTypeSupport.readFloatVector(reader, dimension, vector, offset);
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
                assert neighborCount <= maxDegree : String.format("Node %d neighborCount %d > M %d", node, neighborCount, maxDegree);
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

    @Override
    public String toString() {
        return String.format("OnDiskGraphIndex(size=%d, entryPoint=%d)", size, entryNode);
    }

    /**
     * @param out              the output to write to
     * @param graph            the graph to write
     * @param vectors the vectors associated with each node
     *
     * Writes the graph using the default sequential renumbering.
     */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, DataOutput out) throws IOException {
        write(graph, vectors, getSequentialRenumbering(graph), out);
    }

    /**
     * @param out              the output to write to
     * @param graph            the graph to write
     * @param vectors the vectors associated with each node
     * @param oldToNewOrdinals A map from old to new ordinals. If ordinal numbering does not matter,
     *                         you can use `getSequentialRenumbering`, which will "fill in" holes left by
     *                         any deleted nodes.
     */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, Map<Integer, Integer> oldToNewOrdinals, DataOutput out)
            throws IOException
    {
        var ivw = new InlineVectorsWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, View view, int node) throws IOException {
                vectorTypeSupport.writeFloatVector(out, vectors.getVector(node));
            }
        };

        write(out, graph, __ -> {}, ivw, oldToNewOrdinals);
    }

    public interface InlineVectorsWriter {
        int dimension();

        void write(DataOutput out, View view, int node) throws IOException;
    }

    /**
     * @param out              the output to write to
     * @param graph            the graph to write
     * @param vectorsWriter    knows how to write the vectors associated with each node in the graph
     * @param oldToNewOrdinals A map from old to new ordinals. If ordinal numbering does not matter,
     *                         you can use `getSequentialRenumbering`, which will "fill in" holes left by
     *                         any deleted nodes.
     */
    protected static void write(DataOutput out, GraphIndex graph,
                                Consumer<DataOutput> headerWriter,
                                InlineVectorsWriter vectorsWriter,
                                Map<Integer, Integer> oldToNewOrdinals)
            throws IOException
    {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        if (oldToNewOrdinals.size() != graph.size()) {
            throw new IllegalArgumentException(String.format("ordinalMapper size %d does not match graph size %d",
                                                             oldToNewOrdinals.size(), graph.size()));
        }

        var entriesByNewOrdinal = new ArrayList<>(oldToNewOrdinals.entrySet());
        entriesByNewOrdinal.sort(Comparator.comparingInt(Map.Entry::getValue));
        // the last new ordinal should be size-1
        if (graph.size() > 0 && entriesByNewOrdinal.get(entriesByNewOrdinal.size() - 1).getValue() != graph.size() - 1) {
            throw new IllegalArgumentException("oldToNewOrdinals produced out-of-range entries");
        }

        try (var view = graph.getView()) {
            // graph-level properties
            out.writeInt(graph.size());
            out.writeInt(vectorsWriter.dimension());
            out.writeInt(view.entryNode());
            out.writeInt(graph.maxDegree());
            headerWriter.accept(out);

            // for each graph node, write the associated vector and its neighbors
            for (int i = 0; i < oldToNewOrdinals.size(); i++) {
                var entry = entriesByNewOrdinal.get(i);
                int originalOrdinal = entry.getKey();
                int newOrdinal = entry.getValue();
                if (!graph.containsNode(originalOrdinal)) {
                    continue;
                }

                out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
                vectorsWriter.write(out, view, originalOrdinal);

                var neighbors = view.getNeighborsIterator(originalOrdinal);
                out.writeInt(neighbors.size());
                int n = 0;
                for (; n < neighbors.size(); n++) {
                    out.writeInt(oldToNewOrdinals.get(neighbors.nextInt()));
                }
                assert !neighbors.hasNext();

                // pad out to maxEdgesPerNode
                for (; n < graph.maxDegree(); n++) {
                    out.writeInt(-1);
                }
            }
        } catch (Exception e) {
            throw new IOException(e);
        }
    }
}
