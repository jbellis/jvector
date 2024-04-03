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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.annotations.Experimental;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.pq.QuickADCPQDecoder;
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
 * TODO: Permit maxDegree != 32.
 * TODO: Permit 256 PQ clusters by quantizing floats to one byte.
 */
@Experimental
public class ADCGraphIndex extends OnDiskGraphIndex<ADCGraphIndex.View, ADCGraphIndex.CachedNode>
{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;
    private final ThreadLocal<VectorFloat<?>> results;

    protected ADCGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighborsOffset, ProductQuantization pq)
    {
        super(readerSupplier, info, neighborsOffset);
        this.pq = pq;
        this.results = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(maxDegree));
    }

    public static ADCGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            var pq = ProductQuantization.load(reader);
            long neighborsOffset = reader.getPosition();
            return new ADCGraphIndex(readerSupplier, info, neighborsOffset, pq);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /** return a View that can be safely queried concurrently */
    @Override
    public View getView()
    {
        return new View(readerSupplier.get());
    }

    public class View extends OnDiskView<CachedNode> implements ADCView
    {
        private final ByteSequence<?> packedNeighbors;

        public View(RandomAccessReader reader)
        {
            super(reader, ADCGraphIndex.this);
            this.packedNeighbors = vectorTypeSupport.createByteSequence(maxDegree * pq.getCompressedSize());
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
            throw new UnsupportedOperationException(); // need to copy reader
        }

        // getVector isn't called on the hot path, only getVectorInto, so we don't bother using a shared value
        @Override
        public VectorFloat<?> getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) pq.getCompressedSize() * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            try {
                long diskOffset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) pq.getCompressedSize() * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(diskOffset);
                vectorTypeSupport.readFloatVector(reader, dimension, vector, offset);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        protected long neighborsOffsetFor(int node) {
            return neighborsOffset +
                    (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES + (long) pq.getCompressedSize() * maxDegree) +
                    (node * (long) Integer.BYTES * (maxDegree + 1));
        }

        public ByteSequence<?> getPackedNeighbors(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES)
                        + ((node) * (pq.getCompressedSize() * (long) maxDegree + Integer.BYTES * (long) (maxDegree + 1))));
                vectorTypeSupport.readByteSequence(reader, packedNeighbors);
                return packedNeighbors;
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public VectorFloat<?> reusableResults() {
            return results.get();
        }

        public ProductQuantization getProductQuantization() {
            return pq;
        }

        @Override
        public int size() {
            return ADCGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return ADCGraphIndex.this.entryNode;
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
            return QuickADCPQDecoder.newDecoder(this, query, similarityFunction);
        }

        @Override
        CachedNode loadCachedNode(int node, int[] neighbors) {
            return new CachedNode(neighbors, getVector(node), getPackedNeighbors(node).copy());
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return ScoreFunction.ExactScoreFunction.from(queryVector, vsf, this);
        }

        @Override
        RerankingView cachedWith(GraphCache<CachedNode> cache) {
            return new CachedView(cache, this);
        }
    }

    static class CachedNode extends GraphCache.CachedNode {
        final VectorFloat<?> vector;
        final ByteSequence<?> packedNeighbors;

        public CachedNode(int[] neighbors, VectorFloat<?> vector, ByteSequence<?> packedNeighbors) {
            super(neighbors);
            this.vector = vector;
            this.packedNeighbors = packedNeighbors;
        }
    }

    class CachedView extends CachingGraphIndex.View<View, CachedNode> implements ADCView, RandomAccessVectorValues {
        public CachedView(GraphCache<CachedNode> cache, View view) {
            super(cache, view);
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return ScoreFunction.ExactScoreFunction.from(queryVector, vsf, this);
        }

        @Override
        public int dimension() {
            return view.dimension();
        }

        @Override
        public VectorFloat<?> getVector(int nodeId) {
            var node = getCachedNode(nodeId);
            if (node != null) {
                return node.vector;
            }
            return view.getVector(nodeId);
        }

        @Override
        public void getVectorInto(int nodeId, VectorFloat<?> destinationVector, int offset) {
            var node = (CachedNode) getCachedNode(nodeId);
            if (node != null) {
                destinationVector.copyFrom(node.vector, 0, offset, node.vector.length());
                return;
            }
            view.getVectorInto(nodeId, destinationVector, offset);
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException();
        }

        @Override
        public ByteSequence<?> getPackedNeighbors(int nodeId) {
            var node = (CachedNode) getCachedNode(nodeId);
            if (node != null) {
                return node.packedNeighbors;
            }
            return view.getPackedNeighbors(nodeId);
        }

        @Override
        public VectorFloat<?> reusableResults() {
            return view.reusableResults();
        }

        @Override
        public ProductQuantization getProductQuantization() {
            return view.getProductQuantization();
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
            return QuickADCPQDecoder.newDecoder(this, query, similarityFunction);
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
                pqVectors.getProductQuantization().write(out);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        };

        ByteSequence<?> compressedNeighbors = vectorTypeSupport.createByteSequence(pqVectors.getCompressedSize() * graph.maxDegree());
        var ivw = new InlineWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, GraphIndex.View view, int node) throws IOException {
                vectorTypeSupport.writeFloatVector(out, vectors.getVector(node));

                var neighbors = view.getNeighborsIterator(node);
                int n = 0;
                var neighborSize = neighbors.size();

                compressedNeighbors.zero(); // TODO: make more efficient
                for (; n < neighborSize; n++) {
                    var compressed = pqVectors.get(neighbors.nextInt());
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
