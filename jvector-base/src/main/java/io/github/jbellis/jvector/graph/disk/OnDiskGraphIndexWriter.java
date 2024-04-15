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
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import org.agrona.collections.Int2IntHashMap;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.IntFunction;

/**
 * Write a graph index to disk, for later loading as an OnDiskGraphIndex.
 */
public class OnDiskGraphIndexWriter implements Closeable {
    private final GraphIndex graph;
    private final GraphIndex.View view;
    private final Map<Integer, Integer> oldToNewOrdinals;
    private final int dimension;
    // we don't use Map features but EnumMap is the best way to make sure we don't
    // accidentally introduce an ordering bug in the future
    private final EnumMap<FeatureId, Feature> featureMap;
    private final BufferedRandomAccessWriter out;
    private final long startOffset;

    private OnDiskGraphIndexWriter(BufferedRandomAccessWriter out,
                                   GraphIndex graph,
                                   Map<Integer, Integer> oldToNewOrdinals,
                                   int dimension,
                                   EnumMap<FeatureId, Feature> features)
            throws IOException
    {
        this.graph = graph;
        this.view = graph.getView();
        this.oldToNewOrdinals = oldToNewOrdinals;
        this.dimension = dimension;
        this.featureMap = features;
        this.out = out;
        this.startOffset = out.getFilePointer();
    }

    @Override
    public void close() throws IOException {
        view.close();
    }

    /**
     * Write the inline features of the given ordinal to the output at the correct offset.
     * Nothing else is written (no headers, no edges).
     */
    public void writeInline(BufferedRandomAccessWriter raf,
                            int ordinal,
                            EnumMap<FeatureId, Feature.State> stateMap)
            throws IOException
    {
        var features = featureMap.values();
        int headerBytes = Integer.BYTES // MAGIC
                + Integer.BYTES // featureid bitset
                + CommonHeader.size()
                + features.stream().mapToInt(Feature::headerSize).sum();
        int edgeSize = Integer.BYTES * (1 + graph.maxDegree());
        int inlineBytes = ordinal * (Integer.BYTES + features.stream().mapToInt(Feature::inlineSize).sum() + edgeSize);
        raf.seek(startOffset + headerBytes + inlineBytes + Integer.BYTES);

        for (var writer : features) {
            var state = stateMap.get(writer.id());
            if (state == null) {
                raf.seek(raf.getFilePointer() + writer.inlineSize());
            } else {
                raf.writeBuffered((out) -> writer.writeInline(out, state));
            }
        }
    }

    /**
     * Write the complete index to the given output.  Features that do not have a supplier are assumed
     * to have already been written by calls to writeInline.  The supplier takes node ordinals
     * and returns FeatureState suitable for Feature.writeInline.
     */
    public void write(EnumMap<FeatureId, IntFunction<Feature.State>> featureStateSuppliers) throws IOException
    {
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

        // graph-level properties
        out.seek(startOffset);
        var commonHeader = new CommonHeader(OnDiskGraphIndex.CURRENT_VERSION, graph.size(), dimension, view.entryNode(), graph.maxDegree());
        var header = new Header(commonHeader, featureMap);
        out.writeBuffered(header::write);

        // for each graph node, write the associated vector and its neighbors
        for (int i = 0; i < oldToNewOrdinals.size(); i++) {
            var entry = entriesByNewOrdinal.get(i);
            int originalOrdinal = entry.getKey();
            int newOrdinal = entry.getValue();
            if (!graph.containsNode(originalOrdinal)) {
                continue;
            }

            out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
            for (var feature : featureMap.values()) {
                var supplier = featureStateSuppliers.get(feature.id());
                if (supplier == null) {
                    out.seek(out.getFilePointer() + feature.inlineSize());
                }
                else {
                    out.writeBuffered((out) -> {
                        feature.writeInline(out, supplier.apply(originalOrdinal));
                    });
                }
            }

            var neighbors = view.getNeighborsIterator(originalOrdinal);
            // pad out to maxEdgesPerNode
            out.writeBuffered((out) -> {
                out.writeInt(neighbors.size());
                int n = 0;
                for (; n < neighbors.size(); n++) {
                    out.writeInt(oldToNewOrdinals.get(neighbors.nextInt()));
                }
                assert !neighbors.hasNext();

                // pad out to maxEdgesPerNode
                for (; n < graph.maxDegree(); n++) {
                    out.writeInt(-1);
                }
            });
        }
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

    /** CRC32 checksum of bytes written since the starting offset */
    public long checksum() throws IOException {
        long endOffset = out.getFilePointer();
        return out.checksum(startOffset, endOffset);
    }

    /**
     * Builder for OnDiskGraphIndexWriter, with optional features.
     */
    public static class Builder {
        private final GraphIndex graphIndex;
        private final Map<Integer, Integer> oldToNewOrdinals;
        private final EnumMap<FeatureId, Feature> features;
        private final BufferedRandomAccessWriter out;

        public Builder(GraphIndex graphIndex, BufferedRandomAccessWriter out) {
            this(graphIndex, out, getSequentialRenumbering(graphIndex));
        }

        public Builder(GraphIndex graphIndex, BufferedRandomAccessWriter out, Map<Integer, Integer> oldToNewOrdinals) {
            this.graphIndex = graphIndex;
            this.oldToNewOrdinals = oldToNewOrdinals;
            this.out = out;
            this.features = new EnumMap<>(FeatureId.class);
        }


        public Builder with(Feature feature) {
            features.put(feature.id(), feature);
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

            return new OnDiskGraphIndexWriter(out, graphIndex, oldToNewOrdinals, dimension, features);
        }
    }
}
