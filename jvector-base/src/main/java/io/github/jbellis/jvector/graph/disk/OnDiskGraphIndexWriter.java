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

import io.github.jbellis.jvector.disk.BufferedRandomAccessWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import org.agrona.collections.Int2IntHashMap;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Collection;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Write a graph index to disk, for later loading as an OnDiskGraphIndex.
 * <p>
 * Implements FeatureReader to allow incremental construction of a larger-than-memory graph
 * (using the writer as the source of INLINE_VECTORS or LVQ).
 */
public class OnDiskGraphIndexWriter implements Closeable {
    private final Path outPath;
    private final GraphIndex graph;
    private final GraphIndex.View view;
    private final OrdinalMapper ordinalMapper;
    private final int dimension;
    // we don't use Map features but EnumMap is the best way to make sure we don't
    // accidentally introduce an ordering bug in the future
    private final EnumMap<FeatureId, Feature> featureMap;
    private final BufferedRandomAccessWriter out;
    private final long startOffset;
    private volatile int maxOrdinalWritten;

    private OnDiskGraphIndexWriter(Path outPath,
                                   long startOffset,
                                   GraphIndex graph,
                                   OrdinalMapper oldToNewOrdinals,
                                   int dimension,
                                   EnumMap<FeatureId, Feature> features)
            throws IOException
    {
        this.outPath = outPath;
        this.graph = graph;
        this.view = graph.getView();
        this.ordinalMapper = oldToNewOrdinals;
        this.dimension = dimension;
        this.featureMap = features;
        this.out = new BufferedRandomAccessWriter(outPath);
        this.startOffset = startOffset;
    }

    @Override
    public synchronized void close() throws IOException {
        view.close();
        out.close();
    }

    /**
     * Write the inline features of the given ordinal to the output at the correct offset.
     * Nothing else is written (no headers, no edges).
     */
    public synchronized void writeInline(int ordinal, Map<FeatureId, Feature.State> stateMap) throws IOException
    {
        var features = featureMap.values();
        out.seek(getOffsetForOrdinal(ordinal, features));

        for (var feature : features) {
            var state = stateMap.get(feature.id());
            if (state == null) {
                out.seek(out.getFilePointer() + feature.inlineSize());
            } else {
                feature.writeInline(out, state);
            }
        }

        maxOrdinalWritten = Math.max(maxOrdinalWritten, ordinal);
    }

    public int getMaxOrdinal() {
        return maxOrdinalWritten;
    }

    private long getOffsetForOrdinal(int ordinal, Collection<Feature> features) {
        int headerBytes = Integer.BYTES // MAGIC
                + Integer.BYTES // featureid bitset
                + CommonHeader.size()
                + features.stream().mapToInt(Feature::headerSize).sum();
        int edgeSize = Integer.BYTES * (1 + graph.maxDegree());
        int inlineBytes = ordinal * (Integer.BYTES + features.stream().mapToInt(Feature::inlineSize).sum() + edgeSize);
        return startOffset + headerBytes + inlineBytes + Integer.BYTES;
    }

    @SuppressWarnings("resource")
    public FeatureSource getFeatureSource() {
        RandomAccessFile in;
        try {
            in = new RandomAccessFile(outPath.toFile(), "r");
        } catch (FileNotFoundException e) {
            throw new AssertionError(e);
        }

        return new FeatureSource() {
            private byte[] scratch;

            @Override
            public RandomAccessReader inlineReaderForNode(int ordinal, FeatureId featureId) throws IOException {
                // validation
                if (ordinal > maxOrdinalWritten) {
                    throw new IllegalArgumentException("Ordinal " + ordinal + " has not been written yet");
                }
                var toRead = featureMap.get(featureId);
                if (toRead == null) {
                    throw new IllegalStateException("Feature not present: " + featureId);
                }

                synchronized (OnDiskGraphIndexWriter.this) {
                    out.flush();
                }

                // resize the buffer if necessary
                if (scratch == null || scratch.length < toRead.inlineSize()) {
                    scratch = new byte[toRead.inlineSize()];
                }

                // read the feature
                var features = featureMap.values();
                in.seek(getOffsetForOrdinal(ordinal, features));
                for (var feature : features) {
                    var featureSize = feature.inlineSize();
                    if (feature.id() != featureId) {
                        in.seek(in.getFilePointer() + featureSize);
                        continue;
                    }

                    in.readFully(scratch, 0, featureSize);
                    return new ByteBufferReader(ByteBuffer.wrap(scratch, 0, featureSize));
                }
                throw new AssertionError(); // we checked for the feature in the map at the start
            }

            @Override
            public void close() throws IOException {
                in.close();
            }
        };
    }

    /**
     * Write the complete index to the given output.  Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline.  The supplier takes node ordinals
     * and returns FeatureState suitable for Feature.writeInline.
     */
    public synchronized void write(Map<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        for (var id : featureStateSuppliers.entrySet()) {
            if (!featureMap.containsKey(id.getKey())) {
                throw new IllegalArgumentException("Feature supplier provided for feature not in the graph");
            }
        }

        // graph-level properties
        out.seek(startOffset);
        var graphSize = graph.size();
        var commonHeader = new CommonHeader(OnDiskGraphIndex.CURRENT_VERSION, graphSize, dimension, view.entryNode(), graph.maxDegree());
        var header = new Header(commonHeader, featureMap);
        header.write(out);

        // for each graph node, write the associated vector and its neighbors
        for (int newOrdinal = 0; newOrdinal < graphSize; newOrdinal++) {
            var originalOrdinal = ordinalMapper.newToOld(newOrdinal);
            if (!graph.containsNode(originalOrdinal)) {
                var msg = String.format("Ordinal mapper mapped new ordinal %s to non-existing node %s", newOrdinal, originalOrdinal);
                throw new IllegalStateException(msg);
            }

            out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            for (var feature : featureMap.values()) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    out.seek(out.getFilePointer() + feature.inlineSize());
                }
                else {
                    feature.writeInline(out, supplier.apply(originalOrdinal));
                }
            }

            var neighbors = view.getNeighborsIterator(originalOrdinal);
            // pad out to maxEdgesPerNode
            out.writeInt(neighbors.size());
            int n = 0;
            for (; n < neighbors.size(); n++) {
                var newNeighborOrdinal = ordinalMapper.oldToNew(neighbors.nextInt());
                if (newNeighborOrdinal < 0 || newNeighborOrdinal >= graphSize) {
                    var msg = String.format("Neighbor ordinal out of bounds: %d/%d", newNeighborOrdinal, graphSize);
                    throw new IllegalStateException(msg);
                }
                out.writeInt(newNeighborOrdinal);
            }
            assert !neighbors.hasNext();

            // pad out to maxEdgesPerNode
            for (; n < graph.maxDegree(); n++) {
                out.writeInt(-1);
            }
        }
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.  "Holes" left by
     * deleted nodes are filled in by shifting down the new ordinals.
     */
    public static Map<Integer, Integer> sequentialRenumbering(GraphIndex graph) {
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

    /** CRC32 checksum of bytes written since the starting offset */
    public synchronized long checksum() throws IOException {
        long endOffset = out.getFilePointer();
        return out.checksum(startOffset, endOffset);
    }

    /**
     * Builder for OnDiskGraphIndexWriter, with optional features.
     */
    public static class Builder {
        private final GraphIndex graphIndex;
        private final Path outPath;
        private final EnumMap<FeatureId, Feature> features;
        private OrdinalMapper ordinalMapper;
        private long startOffset;

        public Builder(GraphIndex graphIndex, Path outPath) {
            this.graphIndex = graphIndex;
            this.outPath = outPath;
            this.features = new EnumMap<>(FeatureId.class);
        }

        public Builder with(Feature feature) {
            features.put(feature.id(), feature);
            return this;
        }

        public Builder withMapper(OrdinalMapper ordinalMapper) {
            this.ordinalMapper = ordinalMapper;
            return this;
        }

        public Builder withStartOffset(long startOffset) {
            this.startOffset = startOffset;
            return this;
        }

        public OnDiskGraphIndexWriter build() throws IOException {
            if (features.containsKey(FeatureId.FUSED_ADC) && !(features.containsKey(FeatureId.LVQ) || features.containsKey(FeatureId.INLINE_VECTORS))) {
                throw new IllegalArgumentException("Fused ADC requires an exact score source.");
            }

            int dimension;
            if (features.containsKey(FeatureId.INLINE_VECTORS)) {
                dimension = ((InlineVectors) features.get(FeatureId.INLINE_VECTORS)).dimension();
            } else if (features.containsKey(FeatureId.LVQ)) {
                dimension = ((LVQ) features.get(FeatureId.LVQ)).dimension();
            } else {
                throw new IllegalArgumentException("Either LVQ or inline vectors must be provided.");
            }

            if (ordinalMapper == null) {
                ordinalMapper = new MapMapper(sequentialRenumbering(graphIndex));
            }
            return new OnDiskGraphIndexWriter(outPath, startOffset, graphIndex, ordinalMapper, dimension, features);
        }

        public Builder withMap(Map<Integer, Integer> oldToNewOrdinals) {
            return withMapper(new MapMapper(oldToNewOrdinals));
        }
    }

    public interface OrdinalMapper {
        int oldToNew(int oldOrdinal);
        int newToOld(int newOrdinal);
    }

    public static class IdentityMapper implements OrdinalMapper {
        @Override
        public int oldToNew(int oldOrdinal) {
            return oldOrdinal;
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newOrdinal;
        }
    }

    private static class MapMapper implements OrdinalMapper {
        private final Map<Integer, Integer> oldToNew;
        private final int[] newToOld;

        public MapMapper(Map<Integer, Integer> oldToNew) {
            this.oldToNew = oldToNew;
            this.newToOld = new int[oldToNew.size()];
            oldToNew.forEach((old, newOrdinal) -> newToOld[newOrdinal] = old);
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew.get(oldOrdinal);
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld[newOrdinal];
        }
    }
}
