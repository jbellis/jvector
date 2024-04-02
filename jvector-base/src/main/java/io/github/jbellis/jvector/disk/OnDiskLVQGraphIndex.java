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
import io.github.jbellis.jvector.graph.LVQView;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Map;
import java.util.function.Consumer;

public class OnDiskLVQGraphIndex extends OnDiskGraphIndex
{
    public final LocallyAdaptiveVectorQuantization lvq;

    protected OnDiskLVQGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighorsOffset, LocallyAdaptiveVectorQuantization lvq)
    {
        super(readerSupplier, info, neighorsOffset);
        this.lvq = lvq;
    }

    public static OnDiskLVQGraphIndex load(ReaderSupplier readerSupplier, long offset)
    {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            var lvq = LocallyAdaptiveVectorQuantization.load(reader);
            long neighborsOffset = offset + CommonHeader.SIZE + lvq.serializedSize();
            return new OnDiskLVQGraphIndex(readerSupplier, info, neighborsOffset, lvq);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /** return a Graph that can be safely queried concurrently */
    public OnDiskLVQGraphIndex.OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    private long encodedVectorSize() {
        var base = 2 * Float.BYTES;
        if (dimension % 64 == 0) {
            return dimension + base;
        } else {
            return (dimension / 64 + 1) * 64 + base;
        }
    }

    public class OnDiskView implements LVQView, AutoCloseable
    {
        private final RandomAccessReader reader;
        private final int[] neighbors;
        private final ByteSequence<?> packedVector;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
            this.neighbors = new int[maxDegree];
            this.packedVector = vectorTypeSupport.createByteSequence(new byte[dimension]);
        }

        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + encodedVectorSize() + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(offset);
                var bias = reader.readFloat();
                var scale = reader.readFloat();
                vectorTypeSupport.readByteSequence(reader, packedVector);
                return new LocallyAdaptiveVectorQuantization.PackedVector(packedVector, bias, scale);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public VectorFloat<?> getVector(int node) {
            return null; // TODO HACK
        }

        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            // TODO HACK
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException();
        }

        public NodesIterator getNeighborsIterator(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + encodedVectorSize()) +
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
            return OnDiskLVQGraphIndex.this.size();
        }

        @Override
        public int dimension() {
            return OnDiskLVQGraphIndex.this.dimension;
        }

        @Override
        public int entryNode() {
            return OnDiskLVQGraphIndex.this.entryNode;
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

    @Override // FIXME
    public long ramBytesUsed() {
        return Long.BYTES + 4 * Integer.BYTES;
    }

    @Override
    public String toString() {
        return String.format("OnDiskLVQGraphIndex(size=%d, entryNode=%d)",
                             size, entryNode);
    }

    /**
     * @param graph the graph to write
     * @param vectors the vectors associated with each node (to be quantized by `lvq`)
     * @param lvq the LocallyAdaptiveVectorQuantization to use
     * @param out the output to write to
     *
     * If any nodes have been deleted, you must use the overload specifying `oldToNewOrdinals` instead.
     */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, LocallyAdaptiveVectorQuantization lvq, DataOutput out)
            throws IOException
    {
        Consumer<DataOutput> headerWriter = out_ -> {
            try {
                lvq.write(out_);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        };

        var vectorList = new ArrayList<VectorFloat<?>>(vectors.size());
        for (int i = 0; i < vectors.size(); i++) {
            vectorList.add(vectors.getVector(i));
        }
        var quantizedVectors = lvq.encodeAll(vectorList);
        var vectorsWriter = new InlineVectorsWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, View view, int node) throws IOException {
                quantizedVectors[node].writePacked(out);
            }
        };
        write(out, graph, headerWriter, vectorsWriter, getSequentialRenumbering(graph));
    }
}
