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
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.QuickADCPQDecoder;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.function.Consumer;

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
public class OnDiskADCGraphIndex extends OnDiskGraphIndex
{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    final PQVectors pqv;
    final ThreadLocal<VectorFloat<?>> results;

    protected OnDiskADCGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighborsOffset, PQVectors pqv)
    {
        super(readerSupplier, info, neighborsOffset);
        this.pqv = pqv;
        this.results = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(maxDegree));
    }

    public static OnDiskADCGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            var pqv = PQVectors.load(reader);
            long neighborsOffset = reader.getPosition();
            return new OnDiskADCGraphIndex(readerSupplier, info, neighborsOffset, pqv);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /** return a View that can be safely queried concurrently */
    @Override
    public OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    public class OnDiskView implements ADCView, AutoCloseable
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
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + pqv.getCompressedSize() * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
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
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + pqv.getCompressedSize() * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
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

        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
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
    public String toString() {
        return String.format("OnDiskADCGraphIndex(size=%d, entryPoint=%d)", size, entryNode);
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
        if (pqVectors.getProductQuantization().getClusterCount() != 32) {
            throw new IllegalArgumentException("PQVectors must be generated with a 32-cluster PQ");
        }

        if (graph.maxDegree() != 32) {
            throw new IllegalArgumentException("Graph must be generated with a max degree of 32");
        }

        Consumer<DataOutput> headerWriter = out_ -> {
            try {
                pqVectors.write(out);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        };

        ByteSequence<?> compressedNeighbors = vectorTypeSupport.createByteSequence(pqVectors.getCompressedSize() * graph.maxDegree());
        var ivw = new InlineVectorsWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, View view, int node) throws IOException {
                vectorTypeSupport.writeFloatVector(out, vectors.getVector(node));
                var neighbors = view.getNeighborsIterator(node);
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
            }
        };

        OnDiskGraphIndex.write(out, graph, headerWriter, ivw, getSequentialRenumbering(graph));
    }
}
