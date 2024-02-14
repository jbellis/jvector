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

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.graph.ADCView;
import io.github.jbellis.jvector.graph.ApproximateScoreProvider;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.QuickADCPQDecoder;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2IntHashMap;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Experimental!
 * A GraphIndex that is stored on disk.  This is a read-only index. This index fuses information about the encoded
 * neighboring vectors along with each ordinal, permitting accelerated ADC computation.
 * <p>
 * TODO: Use a limited PQVectors that doesn't load all encoded vectors into memory. These are only used at graph
 * entry points and it's fine to go to disk.
 * TODO: Permit maxDegree != 32.
 * TODO: Permit 256 PQ clusters by quantizing floats to one byte.
 */
@Experimental
public class OnDiskADCGraphIndex implements GraphIndex, AutoCloseable, Accountable
{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ReaderSupplier readerSupplier;
    private final long neighborsOffset;
    private final int size;
    private final int entryNode;
    private final int maxDegree;
    private final int dimension;
    final PQVectors pqv;
    final ThreadLocal<VectorFloat<?>> results;


    public OnDiskADCGraphIndex(ReaderSupplier readerSupplier, long offset)
    {
        this.readerSupplier = readerSupplier;
        this.neighborsOffset = offset + 5 * Integer.BYTES;
        try (var reader = readerSupplier.get()) {
            reader.seek(offset);
            size = reader.readInt();
            dimension = reader.readInt();
            entryNode = reader.readInt();
            maxDegree = reader.readInt();
            var subspaceCount = reader.readInt();
            var pqOffset = offset + 5 * Integer.BYTES + size * (Integer.BYTES + (long) dimension * Float.BYTES
                    + (long) subspaceCount * maxDegree + (long) Integer.BYTES * (maxDegree + 1));
            pqv = PQVectors.load(reader, pqOffset);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskADCGraphIndex at offset " + offset, e);
        }
        results = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(maxDegree));
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

    /** return a View that can be safely queried concurrently */
    public OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    public class OnDiskView implements ADCView, ApproximateScoreProvider, AutoCloseable
    {
        private final RandomAccessReader reader;
        private final int[] neighbors;
        private final ByteSequence<?> packedNeighbors;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
            this.neighbors = new int[maxDegree];
            this.packedNeighbors = vectorTypeSupport.createByteSequence(maxDegree * pqv.getCompressedSize());
        }

        public VectorFloat<?> getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + pqv.getCompressedSize() * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public NodesIterator getNeighborsIterator(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES + pqv.getCompressedSize() * maxDegree) +
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

        public ByteSequence<?> getPackedNeighbors(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES)
                        + ((node) * (pqv.getCompressedSize() * (long) maxDegree + Integer.BYTES * (long) (maxDegree + 1))));
                vectorTypeSupport.readByteSequence(reader, packedNeighbors);
                return packedNeighbors;
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
            return QuickADCPQDecoder.newDecoder(this, query, similarityFunction);
        }

        public VectorFloat<?> reusableResults() {
            return results.get();
        }

        public PQVectors getPQVectors() {
            return pqv;
        }

        @Override
        public int size() {
            return OnDiskADCGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return OnDiskADCGraphIndex.this.entryNode;
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
     * @param out the output to write to
     *
     * If any nodes have been deleted, you must use the overload specifying `oldToNewOrdinals` instead.
     */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, PQVectors pqVectors, DataOutput out)
            throws IOException
    {
        try (var view = graph.getView()) {
            if (view.getIdUpperBound() > graph.size()) {
                throw new IllegalArgumentException("Graph contains deletes, must specify oldToNewOrdinals map");
            }
        } catch (Exception e) {
            throw new IOException(e);
        }
        write(graph, vectors, getSequentialRenumbering(graph), pqVectors, out);
    }

    /**
     * @param graph the graph to write
     * @param vectors the vectors associated with each node
     * @param oldToNewOrdinals A map from old to new ordinals. If ordinal numbering does not matter,
     *                         you can use `getSequentialRenumbering`, which will "fill in" holes left by
     *                         any deleted nodes.
     * @param pqVectors PQVectors generated by encoding `vectors` with a `ProductQuantization`. These
     *                  compressed representations are embedded in the serialized graph to support accelerated ADC.
     * @param out the output to write to
     */
    public static void write(GraphIndex graph,
                                 RandomAccessVectorValues vectors,
                                 Map<Integer, Integer> oldToNewOrdinals,
                                 PQVectors pqVectors,
                                 DataOutput out)
            throws IOException
    {
        if (pqVectors.getProductQuantization().getClusterCount() != 32) {
            throw new IllegalArgumentException("PQVectors must be generated with a 32-cluster PQ");
        }

        if (graph.maxDegree() != 32) {
            throw new IllegalArgumentException("Graph must be generated with a max degree of 32");
        }

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
            out.writeInt(vectors.dimension());
            out.writeInt(view.entryNode());
            out.writeInt(graph.maxDegree());
            out.writeInt(pqVectors.getCompressedSize());
            ByteSequence<?> compressedNeighbors = vectorTypeSupport.createByteSequence(pqVectors.getCompressedSize() * graph.maxDegree());

            // for each graph node, write the associated vector and its neighbors
            for (int i = 0; i < oldToNewOrdinals.size(); i++) {
                var entry = entriesByNewOrdinal.get(i);
                int originalOrdinal = entry.getKey();
                int newOrdinal = entry.getValue();
                if (!graph.containsNode(originalOrdinal)) {
                    continue;
                }

                out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
                vectorTypeSupport.writeFloatVector(out, vectors.vectorValue(originalOrdinal));
                var neighbors = view.getNeighborsIterator(originalOrdinal);
                int n = 0;
                var neighborSize = neighbors.size();

                compressedNeighbors.zero(); // TODO: make more efficient
                for (; n < neighborSize; n++) {
                    var compressed = pqVectors.get(neighbors.next());
                    for (int j = 0; j < pqVectors.getCompressedSize(); j++) {
                        compressedNeighbors.set(j * graph.maxDegree() + n, compressed.get(j));
                    }
                }

                vectorTypeSupport.writeByteSequence(out, compressedNeighbors);

                neighbors = view.getNeighborsIterator(originalOrdinal);
                out.writeInt(neighbors.size());
                n = 0;
                for (; n < neighbors.size(); n++) {
                    out.writeInt(oldToNewOrdinals.get(neighbors.nextInt()));
                }
                assert !neighbors.hasNext();

                // pad out to maxEdgesPerNode
                for (; n < graph.maxDegree(); n++) {
                    out.writeInt(-1);
                }
            }
            pqVectors.write(out);
        } catch (Exception e) {
            throw new IOException(e);
        }
    }
}
