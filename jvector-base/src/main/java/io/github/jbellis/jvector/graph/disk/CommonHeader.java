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
import io.github.jbellis.jvector.graph.GraphIndex;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Base header for OnDiskGraphIndex functionality.
 */
public class CommonHeader {
    private static final int V4_MAX_LAYERS = 32;

    public final int version;
    public final int dimension;
    public final int entryNode;
    public final List<LayerInfo> layerInfo;

    CommonHeader(int version, int dimension, int entryNode, List<LayerInfo> layerInfo) {
        this.version = version;
        this.dimension = dimension;
        this.entryNode = entryNode;
        this.layerInfo = layerInfo;
    }

    public CommonHeader(int version, GraphIndex graph, int entryNode, int dimension) {
        this(version, dimension, entryNode, LayerInfo.fromGraph(graph));
    }

    void write(DataOutput out) throws IOException {
        if (version >= 3) {
            out.writeInt(OnDiskGraphIndex.MAGIC);
            out.writeInt(version);
        }
        out.writeInt(layerInfo.get(0).size);
        out.writeInt(dimension);
        out.writeInt(entryNode);
        out.writeInt(layerInfo.get(0).degree);
        if (version >= 4) {
            if (layerInfo.size() > V4_MAX_LAYERS) {
                var msg = String.format("Number of layers %d exceeds maximum of %d", layerInfo.size(), V4_MAX_LAYERS);
                throw new IllegalArgumentException(msg);
            }
            out.writeInt(layerInfo.size());
            // Write actual layer info
            for (LayerInfo info : layerInfo) {
                out.writeInt(info.size);
                out.writeInt(info.degree);
            }
            // Pad remaining entries with zeros
            for (int i = layerInfo.size(); i < V4_MAX_LAYERS; i++) {
                out.writeInt(0); // size
                out.writeInt(0); // degree
            }
        } else {
            if (layerInfo.size() > 1) {
                throw new IllegalArgumentException("Layer info is not supported in version " + version);
            }
        }
    }

    static CommonHeader load(RandomAccessReader reader) throws IOException {
        int maybeMagic = reader.readInt();
        int version;
        int size;
        if (maybeMagic == OnDiskGraphIndex.MAGIC) {
            version = reader.readInt();
            size = reader.readInt();
        } else {
            version = 2;
            size = maybeMagic;
        }
        int dimension = reader.readInt();
        int entryNode = reader.readInt();
        int maxDegree = reader.readInt();
        List<LayerInfo> layerInfo;
        if (version < 4) {
            layerInfo = List.of(new LayerInfo(size, maxDegree));
        } else {
            int numLayers = reader.readInt();
            layerInfo = new ArrayList<>();
            for (int i = 0; i < numLayers; i++) {
                LayerInfo info = new LayerInfo(reader.readInt(), reader.readInt());
                layerInfo.add(info);
            }
            // Skip over remaining padding entries
            for (int i = numLayers; i < V4_MAX_LAYERS; i++) {
                reader.readInt();
                reader.readInt();
            }
        }

        return new CommonHeader(version, dimension, entryNode, layerInfo);
    }

    int size() {
        return ((version >= 3 ? 2 : 0) // v3: version + magic
                + 4 // v2 fields
                + (version >= 4 ? 1 + 2 * V4_MAX_LAYERS : 0)) // v4: layerinfo count + contents
                * Integer.BYTES;
    }

    @VisibleForTesting
    public static class LayerInfo {
        public final int size;
        public final int degree;

        public LayerInfo(int size, int degree) {
            this.size = size;
            this.degree = degree;
        }

        public static List<LayerInfo> fromGraph(GraphIndex graph) {
            return IntStream.range(0, graph.getMaxLevel() + 1)
                    .mapToObj(i -> new LayerInfo(graph.size(i), graph.getDegree(i)))
                    .collect(Collectors.toList());
        }

        @Override
        public String toString() {
            return "LayerInfo{" +
                    "size=" + size +
                    ", degree=" + degree +
                    '}';
        }

        @Override
        public int hashCode() {
            return Objects.hash(size, degree);
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            LayerInfo other = (LayerInfo) obj;
            return size == other.size && degree == other.degree;
        }
    }
}
