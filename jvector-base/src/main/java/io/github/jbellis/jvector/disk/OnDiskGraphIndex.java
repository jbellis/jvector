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
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Accountable;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

public class OnDiskGraphIndex<T> implements GraphIndex<T>, AutoCloseable, Accountable
{
    private final ReaderSupplier readerSupplier;
    private final long neighborsOffset;
    private final int size;
    private final int entryNode;
    private final int M;
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
            M = reader.readInt();
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int maxEdgesPerNode() {
        return M;
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

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
        }

        public T getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (M + 1)) // earlier entries
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
                        (node * (long) Integer.BYTES * (M + 1)));
                int neighborCount = reader.readInt();
                assert neighborCount <= M : String.format("neighborCount %d > M %d", neighborCount, M);
                return new NodesIterator(neighborCount)
                {
                    int currentNeighborsRead = 0;

                    @Override
                    public int nextInt() {
                        currentNeighborsRead++;
                        try {
                            int ordinal = reader.readInt();
                            assert ordinal <= OnDiskGraphIndex.this.size : String.format("ordinal %d > size %d", ordinal, size);
                            return ordinal;
                        }
                        catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    }

                    @Override
                    public boolean hasNext() {
                        return currentNeighborsRead < neighborCount;
                    }
                };
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
        public void close() throws IOException {
            reader.close();
        }
    }

    @Override
    public NodesIterator getNodes()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public long ramBytesUsed() {
        return Long.BYTES + 4 * Integer.BYTES;
    }

    public void close() throws IOException {
        readerSupplier.close();
    }

    // takes Graph and Vectors separately since I'm reluctant to introduce a Vectors reference
    // to OnHeapGraphIndex just for this method.  Maybe that will end up the best solution,
    // but I'm not sure yet.
    public static <T> void write(GraphIndex<T> graph, RandomAccessVectorValues<T> vectors, DataOutput out) throws IOException {
        assert graph.size() == vectors.size() : String.format("graph size %d != vectors size %d", graph.size(), vectors.size());

        var view = graph.getView();

        // graph-level properties
        out.writeInt(graph.size());
        out.writeInt(vectors.dimension());
        out.writeInt(view.entryNode());
        out.writeInt(graph.maxEdgesPerNode());

        // for each graph node, write the associated vector and its neighbors
        for (int node = 0; node < graph.size(); node++) {
            out.writeInt(node); // unnecessary, but a reasonable sanity check
            Io.writeFloats(out, (float[]) vectors.vectorValue(node));

            var neighbors = view.getNeighborsIterator(node);
            out.writeInt(neighbors.size());
            int n = 0;
            for ( ; n < neighbors.size(); n++) {
                out.writeInt(neighbors.nextInt());
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for ( ; n < graph.maxEdgesPerNode(); n++) {
                out.writeInt(-1);
            }
        }
    }
}
