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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A class representing a graph index stored on disk. The base graph contains only graph structure.
 * <p> * The base graph

 * This graph may be extended with additional features, which are stored inline in the graph and in headers.
 * At runtime, this class may choose the best way to use these features.
 */
public class OnDiskGraphIndex implements GraphIndex, AutoCloseable, Accountable
{
    public static final int CURRENT_VERSION = 4;
    static final int MAGIC = 0xFFFF0D61; // FFFF to distinguish from old graphs, which should never start with a negative size "ODGI"
    static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    final ReaderSupplier readerSupplier;
    final int version;
    final int size;
    final int maxDegree;
    final int dimension;
    final int entryNode;
    final int idUpperBound;
    final int inlineBlockSize; // total size of all inline elements contributed by features
    final EnumMap<FeatureId, ? extends Feature> features;
    final EnumMap<FeatureId, Integer> inlineOffsets;
    private final long neighborsOffset;

    OnDiskGraphIndex(ReaderSupplier readerSupplier, Header header, long neighborsOffset)
    {
        this.readerSupplier = readerSupplier;
        this.version = header.common.version;
        this.size = header.common.size;
        this.dimension = header.common.dimension;
        this.entryNode = header.common.entryNode;
        this.maxDegree = header.common.maxDegree;
        this.idUpperBound = header.common.idUpperBound;
        this.features = header.features;
        this.neighborsOffset = neighborsOffset;
        var inlineBlockSize = 0;
        inlineOffsets = new EnumMap<>(FeatureId.class);
        for (var entry : features.entrySet()) {
            inlineOffsets.put(entry.getKey(), inlineBlockSize);
            inlineBlockSize += entry.getValue().inlineSize();
        }
        this.inlineBlockSize = inlineBlockSize;
    }

    /**
     * Load an index from the given reader supplier, where the index starts at `offset`.
     */
    public static OnDiskGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (var reader = readerSupplier.get()) {
            var header = Header.load(reader, offset);
            return new OnDiskGraphIndex(readerSupplier, header, reader.getPosition());
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /**
     * Load an index from the given reader supplier at offset 0.
     */
    public static OnDiskGraphIndex load(ReaderSupplier readerSupplier) {
        return load(readerSupplier, 0);
    }

    public Set<FeatureId> getFeatureSet() {
        return features.keySet();
    }

    public int getDimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    @Override
    public int getIdUpperBound() {
        return idUpperBound;
    }

    @Override
    public NodesIterator getNodes() {
        try (var reader = readerSupplier.get()) {
            int[] valid_nodes = new int[size];
            int pos = 0;
            for (int node = 0; node < getIdUpperBound(); node++) {
                long node_offset = neighborsOffset +
                        (node * ((long) Integer.BYTES // ids
                                + inlineBlockSize // inline elements
                                + (Integer.BYTES * (long) (maxDegree + 1)) // neighbor count + neighbors)
                        ));
                reader.seek(node_offset);
                if (reader.readInt() != -1) {
                    valid_nodes[pos++] = node;
                }
            }
            return new NodesIterator.ArrayNodesIterator(valid_nodes, size);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public long ramBytesUsed() {
        return Long.BYTES + 6 * Integer.BYTES + RamUsageEstimator.NUM_BYTES_OBJECT_REF
                + (long) 2 * RamUsageEstimator.NUM_BYTES_OBJECT_REF * FeatureId.values().length;
    }

    public void close() throws IOException {
        // caller is responsible for closing ReaderSupplier
    }

    @Override
    public String toString() {
        return String.format("OnDiskGraphIndex(size=%d, entryPoint=%d, features=%s)", size, entryNode,
                features.keySet().stream().map(Enum::name).collect(Collectors.joining(",")));
    }

    // re-declared to specify type
    @Override
    public View getView() {
        try {
            return new View(readerSupplier.get());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public class View implements FeatureSource, ScoringView, RandomAccessVectorValues {
        protected final RandomAccessReader reader;
        private final int[] neighbors;

        public View(RandomAccessReader reader) {
            this.reader = reader;
            this.neighbors = new int[maxDegree];
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

        protected long inlineOffsetFor(int node, FeatureId featureId) {
            return neighborsOffset +
                    (node * ((long) Integer.BYTES // ids
                            + inlineBlockSize // inline elements
                            + (Integer.BYTES * (long) (maxDegree + 1)) // neighbor count + neighbors)
                    )) + Integer.BYTES + // id
                    inlineOffsets.get(featureId);
        }

        long neighborsOffsetFor(int node) {
            return neighborsOffset +
                    (node + 1) * (Integer.BYTES + (long) inlineBlockSize) +
                    (node * (long) Integer.BYTES * (maxDegree + 1));
        }

        @Override
        public RandomAccessReader inlineReaderForNode(int node, FeatureId featureId) throws IOException {
            long offset = inlineOffsetFor(node, featureId);
            reader.seek(offset);
            return reader;
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            if (!features.containsKey(FeatureId.INLINE_VECTORS)) {
                throw new UnsupportedOperationException("No inline vectors in this graph");
            }

            try {
                long offset = inlineOffsetFor(node, FeatureId.INLINE_VECTORS);
                reader.seek(offset);
                return vectorTypeSupport.readFloatVector(reader, dimension);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            if (!features.containsKey(FeatureId.INLINE_VECTORS)) {
                throw new UnsupportedOperationException("No inline vectors in this graph");
            }

            try {
                long diskOffset = inlineOffsetFor(node, FeatureId.INLINE_VECTORS);
                reader.seek(diskOffset);
                vectorTypeSupport.readFloatVector(reader, dimension, vector, offset);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public NodesIterator getNeighborsIterator(int node) {
            try {
                reader.seek(neighborsOffsetFor(node));
                int neighborCount = reader.readInt();
                assert neighborCount <= maxDegree : String.format("Node %d neighborCount %d > M %d", node, neighborCount, maxDegree);
                reader.read(neighbors, 0, neighborCount);
                return new NodesIterator.ArrayNodesIterator(neighbors, neighborCount);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
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
        public int getIdUpperBound() {
            return idUpperBound;
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }


        @Override
        public void close() throws IOException {
            reader.close();
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            if (features.containsKey(FeatureId.INLINE_VECTORS)) {
                return RandomAccessVectorValues.super.rerankerFor(queryVector, vsf);
            } else if (features.containsKey(FeatureId.NVQ_VECTORS)) {
                return ((NVQ) features.get(FeatureId.NVQ_VECTORS)).rerankerFor(queryVector, vsf, this);
            } else {
                throw new UnsupportedOperationException("No reranker available for this graph");
            }
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            if (features.containsKey(FeatureId.FUSED_ADC)) {
                return ((FusedADC) features.get(FeatureId.FUSED_ADC)).approximateScoreFunctionFor(queryVector, vsf, this, rerankerFor(queryVector, vsf));
            } else {
                throw new UnsupportedOperationException("No approximate score function available for this graph");
            }
        }
    }

    /** Convenience function for writing a vanilla DiskANN-style index with no extra Features. */
    public static void write(GraphIndex graph, RandomAccessVectorValues vectors, Path path) throws IOException {
        write(graph, vectors, OnDiskGraphIndexWriter.sequentialRenumbering(graph), path);
    }

    /** Convenience function for writing a vanilla DiskANN-style index with no extra Features. */
    public static void write(GraphIndex graph,
                             RandomAccessVectorValues vectors,
                             Map<Integer, Integer> oldToNewOrdinals,
                             Path path)
            throws IOException
    {
        try (var writer = new OnDiskGraphIndexWriter.Builder(graph, path).withMap(oldToNewOrdinals)
                .with(new InlineVectors(vectors.dimension()))
                .build())
        {
            var suppliers = Feature.singleStateFactory(FeatureId.INLINE_VECTORS,
                                                       nodeId -> new InlineVectors.State(vectors.getVector(nodeId)));
            writer.write(suppliers);
        }
    }

    @VisibleForTesting
    static boolean areHeadersEqual(OnDiskGraphIndex g1, OnDiskGraphIndex g2) {
        return g1.version == g2.version &&
               g1.size == g2.size &&
               g1.maxDegree == g2.maxDegree &&
               g1.dimension == g2.dimension &&
               g1.entryNode == g2.entryNode;
    }
}
