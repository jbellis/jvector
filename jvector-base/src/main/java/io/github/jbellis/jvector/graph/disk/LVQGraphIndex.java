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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.function.Consumer;

public class LVQGraphIndex extends OnDiskGraphIndex<LVQGraphIndex.View, LVQGraphIndex.CachedNode>
{
    public final LocallyAdaptiveVectorQuantization lvq;

    protected LVQGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighborsOffset, LocallyAdaptiveVectorQuantization lvq)
    {
        super(readerSupplier, info, neighborsOffset);
        this.lvq = lvq;
    }

    public static LVQGraphIndex load(ReaderSupplier readerSupplier, long offset)
    {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            var lvq = LocallyAdaptiveVectorQuantization.load(reader);
            long neighborsOffset = offset + CommonHeader.SIZE + lvq.serializedSize();
            return new LVQGraphIndex(readerSupplier, info, neighborsOffset, lvq);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /** return a Graph that can be safely queried concurrently */
    public View getView()
    {
        return new View(readerSupplier.get());
    }

    private long encodedVectorSize() {
        var base = 2 * Float.BYTES;
        if (dimension % 64 == 0) {
            return dimension + base;
        } else {
            return (dimension / 64 + 1) * 64 + base;
        }
    }

    public class View extends OnDiskView<CachedNode> implements LVQView
    {
        private final ByteSequence<?> packedVector;

        public View(RandomAccessReader reader)
        {
            super(reader, LVQGraphIndex.this);
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

        @Override
        protected long neighborsOffsetFor(int node) {
            return neighborsOffset +
                    (node + 1) * (Integer.BYTES + encodedVectorSize()) +
                    (node * (long) Integer.BYTES * (maxDegree + 1));
        }

        @Override
        CachedNode loadCachedNode(int node, int[] neighbors) {
            return new CachedNode(neighbors, getPackedVector(node).copy());
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return lvq.scoreFunctionFrom(queryVector, vsf, this);
        }

        @Override
        RerankingView cachedWith(GraphCache<CachedNode> cache) {
            return new CachedView(cache, this);
        }
    }

    static class CachedNode extends GraphCache.CachedNode {
        final LocallyAdaptiveVectorQuantization.PackedVector packedVector;

        public CachedNode(int[] neighbors, LocallyAdaptiveVectorQuantization.PackedVector packedVector) {
            super(neighbors);
            this.packedVector = packedVector;
        }
    }

    class CachedView extends CachingGraphIndex.View<View, CachedNode> implements LVQView {
        public CachedView(GraphCache<CachedNode> cache, View view) {
            super(cache, view);
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return lvq.scoreFunctionFrom(queryVector, vsf, this);
        }

        @Override
        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int ordinal) {
            var node = getCachedNode(ordinal);
            if (node != null) {
                return node.packedVector;
            }
            return view.getPackedVector(ordinal);
        }
    }

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
        var vectorsWriter = new InlineWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, GraphIndex.View view, int node) throws IOException {
                quantizedVectors[node].writePacked(out);
            }
        };
        write(out, graph, headerWriter, vectorsWriter, getSequentialRenumbering(graph));
    }
}
