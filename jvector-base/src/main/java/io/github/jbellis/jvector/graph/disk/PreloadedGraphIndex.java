package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization.PackedVector;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.util.EnumMap;
import java.util.Set;
import java.util.stream.IntStream;

public class PreloadedGraphIndex implements GraphIndex, AutoCloseable, Accountable {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int version;
    private final int size;
    private final int maxDegree;
    private final int dimension;
    private final int entryNode;
    private final EnumMap<FeatureId, ? extends Feature> features;
    private final int[] neighbors;
    private final int[] neighborCounts;
    private final LVQPackedVectors packedVectors;

    private PreloadedGraphIndex(Header header, int[] neighbors, int[] neighborCounts, PackedVector[] vectors) {
        this.version = header.common.version;
        this.size = header.common.size;
        this.dimension = header.common.dimension;
        this.entryNode = header.common.entryNode;
        this.maxDegree = header.common.maxDegree;
        this.features = header.features;
        this.neighbors = neighbors;
        this.neighborCounts = neighborCounts;
        this.packedVectors = ordinal -> vectors[ordinal];
    }

    public static PreloadedGraphIndex load(ReaderSupplier readerSupplier, long offset) {
        try (RandomAccessReader reader = readerSupplier.get()) {
            Header header = Header.load(reader, offset);
            int size = header.common.size;
            int maxDegree = header.common.maxDegree;

            assert header.features.size() == 1 && header.features.containsKey(FeatureId.LVQ) : "Only LVQ feature must be present";
            LVQ lvq = (LVQ) header.features.get(FeatureId.LVQ);

            int[] neighbors = new int[size * maxDegree];
            int[] neighborCounts = new int[size];
            PackedVector[] vectors = new PackedVector[size];

            reader.seek(reader.getPosition()); // Move to the start of node data

            for (int i = 0; i < size; i++) {
                // Read node ordinal (for sanity check)
                int nodeOrdinal = reader.readInt();
                assert nodeOrdinal == i : "Node ordinal mismatch";

                // Read LVQ feature data
                float bias = reader.readFloat();
                float scale = reader.readFloat();
                ByteSequence<?> bytes = vectorTypeSupport.createByteSequence(lvq.inlineSize() - 2 * Float.BYTES);
                vectorTypeSupport.readByteSequence(reader, bytes);
                vectors[i] = new PackedVector(bytes, bias, scale);

                // Read edge list
                int count = reader.readInt();
                neighborCounts[i] = count;
                reader.read(neighbors, i * maxDegree, count);
                reader.seek(reader.getPosition() + (long) (maxDegree - count) * Integer.BYTES); // Skip unused neighbors
                // set unused neighbor entries to -1 as an additional safeguard
                for (int j = count; j < maxDegree; j++) {
                    neighbors[i * maxDegree + j] = -1;
                }
            }

            return new PreloadedGraphIndex(header, neighbors, neighborCounts, vectors);
        } catch (IOException e) {
            throw new RuntimeException("Error loading PreloadedGraphIndex", e);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public NodesIterator getNodes() {
        return NodesIterator.fromPrimitiveIterator(IntStream.range(0, size).iterator(), size);
    }

    @Override
    public View getView() {
        return new PreloadedView();
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    @Override
    public long ramBytesUsed() {
        throw new UnsupportedOperationException(); // TODO
    }

    @Override
    public void close() {
        // No resources to close
    }

    public Set<FeatureId> getFeatureSet() {
        return features.keySet();
    }

    public class PreloadedView implements GraphIndex.ScoringView {
        @Override
        public NodesIterator getNeighborsIterator(int node) {
            int start = node * maxDegree;
            return new NodesIterator.ArrayNodesIterator(neighbors, start, neighborCounts[node]);
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
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public void close() {
            // No resources to close
        }

        @Override
        public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            LVQ lvq = (LVQ) features.get(FeatureId.LVQ);
            return lvq.rerankerFor(queryVector, vsf, packedVectors);
        }

        @Override
        public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf) {
            throw new UnsupportedOperationException("PreloadedGraphIndex does not support approximate scoring");
        }
    }
}
