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

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.LVQView;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

public class LocallyAdaptiveVectorQuantization implements VectorCompressor<LocallyAdaptiveVectorQuantization.QuantizedVector>{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public final VectorFloat<?> globalMean;

    public LocallyAdaptiveVectorQuantization(VectorFloat<?> globalMean) {
        this.globalMean = globalMean;
    }

    public static LocallyAdaptiveVectorQuantization compute(RandomAccessVectorValues ravv) {
        var ravvCopy = ravv.threadLocalSupplier().get();
        // convert ravvCopy to list
        var list = new ArrayList<VectorFloat<?>>(ravvCopy.size());
        for (int i = 0; i < ravvCopy.size(); i++) {
            list.add(ravvCopy.vectorValue(i));
        }
        return new LocallyAdaptiveVectorQuantization(KMeansPlusPlusClusterer.centroidOf(list));
    }
    @Override
    public QuantizedVector[] encodeAll(List<VectorFloat<?>> vectors, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> vectors.stream().parallel().map(this::encode).toArray(QuantizedVector[]::new)).join();
    }

    @Override
    public QuantizedVector encode(VectorFloat<?> v) {
        // first, subtract the global mean
        var vCentered = VectorUtil.sub(v, globalMean);
        var u = VectorUtil.max(vCentered);
        var l = VectorUtil.min(vCentered);
        // scalar quantization of each component of v centered using a byte, where 0 = l and 255 = u
        var range = u - l;
        var quantized = vectorTypeSupport.createByteSequence(vCentered.length());
        for (int i = 0; i < vCentered.length(); i++) {
            quantized.set(i, quantizeFloatToByte(vCentered.get(i), l, u));
        }
        return new QuantizedVector(quantized, l, (u - l) / 255);
    }

    private static byte quantizeFloatToByte(float value, float minFloat, float maxFloat) {
        // Calculate the quantization step delta
        float delta = (maxFloat - minFloat) / 255;

        // Apply the quantization formula
        int quantizedValue = Math.round((value - minFloat) / delta);

        // Ensure the quantized value is within the 0 to 255 range
        if (quantizedValue < 0) quantizedValue = 0;
        if (quantizedValue > 255) quantizedValue = 255;

        return (byte) quantizedValue;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(globalMean.length());
        vectorTypeSupport.writeFloatVector(out, globalMean);
    }

    public int serializedSize() {
        return Integer.BYTES + globalMean.length() * Float.BYTES;
    }

    private NodeSimilarity.Reranker dotProductRerankerFrom(VectorFloat<?> query, LVQView view) {
        var querySum = VectorUtil.sum(query);
        var queryGlobalBias = VectorUtil.dotProduct(query, globalMean);
        return (nodes, scores) -> {
            var nodeCount = nodes.length;
            for (int i = 0; i < nodeCount; i++) {
                var node = nodes[i];
                var vector = view.getPackedVector(node);
                var lvqDot = VectorUtil.lvqDotProduct(query, vector, querySum);
                lvqDot = lvqDot + queryGlobalBias;
                scores.set(i, (1 + lvqDot) / 2);
            }
        };
    }

    private NodeSimilarity.Reranker euclideanRerankerFrom(VectorFloat<?> query, LVQView view) {
        var shiftedQuery = VectorUtil.sub(query, globalMean);
        return (nodes, scores) -> {
            var nodeCount = nodes.length;
            for (int i = 0; i < nodeCount; i++) {
                var node = nodes[i];
                var vector = view.getPackedVector(node);
                var lvqDist = VectorUtil.lvqSquareL2Distance(shiftedQuery, vector);
                scores.set(i, 1 / (1 + lvqDist));
            }
        };
    }

    public NodeSimilarity.Reranker rerankerFrom(VectorFloat<?> query, VectorSimilarityFunction similarityFunction, LVQView view) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return dotProductRerankerFrom(query, view);
            case EUCLIDEAN:
                return euclideanRerankerFrom(query, view);
            default:
                throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
        }
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return null; // TODO
    }

    public static LocallyAdaptiveVectorQuantization load(RandomAccessReader in) throws IOException {
        int length = in.readInt();
        VectorFloat<?> globalMean = vectorTypeSupport.readFloatVector(in, length);
        return new LocallyAdaptiveVectorQuantization(globalMean);
    }

    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LocallyAdaptiveVectorQuantization that = (LocallyAdaptiveVectorQuantization) o;

        return globalMean.equals(that.globalMean);
    }

    public static class QuantizedVector {
        private final ByteSequence<?> encodedVector;
        private final float bias;
        private final float scale;

        public QuantizedVector(ByteSequence<?> encodedVector, float bias, float scale) {
            this.encodedVector = encodedVector;
            this.bias = bias;
            this.scale = scale;
        }

        public void writePacked(DataOutput out) throws IOException {
            // write the min and max
            // write the encoded vector
            out.writeFloat(bias);
            out.writeFloat(scale);
            int i = 0;
            int mainBlockCount = (encodedVector.length() - (encodedVector.length() % 64)) / 64;
            for (i = 0; i < mainBlockCount; i++) {
                var startIndex = i * 64;
                for (int j = startIndex; j < startIndex + 16; j++) {
                    out.writeByte(encodedVector.get(j));
                    out.writeByte(encodedVector.get(j + 16));
                    out.writeByte(encodedVector.get(j + 32));
                    out.writeByte(encodedVector.get(j + 48));
                }
            }
            var startIndex = i * 64;
            if (startIndex < encodedVector.length()) {
                for (int j = startIndex; j < startIndex + 16; j++) {
                    if (j < encodedVector.length()) {
                        out.writeByte(encodedVector.get(j));
                    } else {
                        out.writeByte(0);
                    }
                    if (j + 16 < encodedVector.length()) {
                        out.writeByte(encodedVector.get(j + 16));
                    } else {
                        out.writeByte(0);
                    }
                    if (j + 32 < encodedVector.length()) {
                        out.writeByte(encodedVector.get(j + 32));
                    } else {
                        out.writeByte(0);
                    }
                    if (j + 48 < encodedVector.length()) {
                        out.writeByte(encodedVector.get(j + 48));
                    } else {
                        out.writeByte(0);
                    }
                }
            }
        }
    }

    public static class PackedVector {
        public final ByteSequence<?> packedVector;
        public final float bias;
        public final float scale;

        public PackedVector(ByteSequence<?> packedVector, float bias, float scale) {
            this.packedVector = packedVector;
            this.bias = bias;
            this.scale = scale;
        }

        public int getQuantized(int index) {
            var blockId = index / 64; // 64 bytes per block
            var inBlockId = index % 64; // switch to an in-block index
            var laneId = inBlockId % 16; // 4 bytes per lane
            var laneOffset = inBlockId / 16; // 16 lanes per block
            var packedIndex = blockId * 64 + laneId * 4 + laneOffset;
            return Byte.toUnsignedInt(packedVector.get(packedIndex));
        }

        public float getDequantized(int index) {
            return (getQuantized(index) * scale) + bias;
        }
    }
}
