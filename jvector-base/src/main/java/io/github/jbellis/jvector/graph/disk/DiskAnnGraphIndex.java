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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Map;

/**
 * Vanilla DiskANN index.  Includes full-precision vectors with edge lists.
 */
public class DiskAnnGraphIndex extends OnDiskGraphIndex<DiskAnnGraphIndex.View, DiskAnnGraphIndex.CachedNode> {
    protected DiskAnnGraphIndex(ReaderSupplier readerSupplier, CommonHeader info, long neighborsOffset) {
        super(readerSupplier, info, neighborsOffset);
    }

    public static DiskAnnGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            var info = CommonHeader.load(reader, offset);
            return new DiskAnnGraphIndex(readerSupplier, info, offset + 4 * Integer.BYTES);
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    @Override
    public View getView() {
        return new View(readerSupplier.get());
    }

    public class View extends OnDiskView<DiskAnnGraphIndex.CachedNode> implements RandomAccessVectorValues {
        public View(RandomAccessReader reader) {
            super(reader, DiskAnnGraphIndex.this);
        }

        @Override
        public int dimension() {
            return dimension;
        }

        // getVector isn't called on the hot path, only getVectorInto, so we don't bother using a shared value
        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            throw new UnsupportedOperationException(); // need to copy reader
        }

        @Override
        protected long neighborsOffsetFor(int node) {
            return neighborsOffset +
                    (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES) +
                    (node * (long) Integer.BYTES * (maxDegree + 1));
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            try {
                long diskOffset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                reader.seek(diskOffset);
                vectorTypeSupport.readFloatVector(reader, dimension, vector, offset);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return ScoreFunction.ExactScoreFunction.from(queryVector, vsf, this);
        }

        @Override
        CachedNode loadCachedNode(int node, int[] neighbors) {
            return new CachedNode(neighbors, getVector(node));
        }

        @Override
        RerankingView cachedWith(GraphCache<CachedNode> cache) {
            return new CachedView(cache, this);
        }
    }

    static class CachedNode extends GraphCache.CachedNode {
        final VectorFloat<?> vector;

        public CachedNode(int[] neighbors, VectorFloat<?> vector) {
            super(neighbors);
            this.vector = vector;
        }
    }

    class CachedView extends CachingGraphIndex.View<View, CachedNode> implements RandomAccessVectorValues {
        public CachedView(GraphCache<CachedNode> cache, View view) {
            super(cache, view);
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
            var node = getCachedNode(nodeId);
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
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            return ScoreFunction.ExactScoreFunction.from(queryVector, vsf, this);
        }
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
        var ivw = new InlineWriter() {
            @Override
            public int dimension() {
                return vectors.dimension();
            }

            @Override
            public void write(DataOutput out, GraphIndex.View view, int node) throws IOException {
                vectorTypeSupport.writeFloatVector(out, vectors.getVector(node));
            }
        };

        write(out, graph, __ -> {}, ivw, oldToNewOrdinals);
    }
}

