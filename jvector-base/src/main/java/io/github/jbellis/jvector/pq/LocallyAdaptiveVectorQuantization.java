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
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.LVQPackedVectors;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
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

/**
 * Implements Locally-Adaptive Vector Quantization (LVQ) as described in
 * "Similarity search in the blink of an eye with compressed indices" (https://arxiv.org/abs/2304.04759).
 * In particular, single-level LVQ-8 is used. To encode, a vector is first de-meaned by subtracting the global mean.
 * Then, each component is quantized using a byte, where 0 = min and 255 = max. The bias and scale are stored for each
 * component to allow for dequantization. Vectors are packed using Turbo LVQ as described in
 * "Locally-Adaptive Quantization for Streaming Vector Search" (https://arxiv.org/pdf/2402.02044.pdf).
 */
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
        var quantized = vectorTypeSupport.createByteSequence(vCentered.length());
        for (int i = 0; i < vCentered.length(); i++) {
            quantized.set(i, quantizeFloatToByte(vCentered.get(i), l, u));
        }
        return new QuantizedVector(quantized, l, (u - l) / 255);
    }

    /**
     * Quantize a float value to an unsigned byte value using the given min and max values.
     * The returned value is a signed byte, so it is in the range -128 to 127. This requires correction before use.
     */
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

    private ScoreFunction.ExactScoreFunction dotProductScoreFunctionFrom(VectorFloat<?> query, LVQPackedVectors packedVectors) {
        /* The query vector (full resolution) will be compared to LVQ quantized vectors that were first de-meaned
         * by subtracting the global mean. The dot product is calculated between the query and the quantized vector.
         * This is <query, quantized + globalMean> = <query, quantized> + <query, globalMean> = <query, quantized> + globalBias.
         * Global bias can be precomputed. For <query, quantized>, we can break this down to <query, byteSequence * vector.scale + broadcast(vector.bias)>.
        * This can be further broken down to vector.scale * <query, byteSequence> + <query, broadcast(vector.bias)>.
        * Since vector.bias is scalar, <query, broadcast(vector.bias)> = vector.bias * <query, broadcast(1)> = vector.bias * querySum.
         */
        var querySum = VectorUtil.sum(query);
        var queryGlobalBias = VectorUtil.dotProduct(query, globalMean);
        return new ScoreFunction.ExactScoreFunction() {
            @Override
            public VectorFloat<?> similarityTo(int[] nodes) {
                var results = vts.createFloatVector(nodes.length);
                var nodeCount = nodes.length;
                for (int i = 0; i < nodeCount; i++) {
                    var node = nodes[i];
                    var vector = packedVectors.getPackedVector(node);
                    var lvqDot = VectorUtil.lvqDotProduct(query, vector, querySum);
                    lvqDot = lvqDot + queryGlobalBias;
                    results.set(i, (1 + lvqDot) / 2);
                }
                return results;
            }

            @Override
            public float similarityTo(int node2) {
                var vector = packedVectors.getPackedVector(node2);
                var lvqDot = VectorUtil.lvqDotProduct(query, vector, querySum);
                lvqDot = lvqDot + queryGlobalBias;
                return (1 + lvqDot) / 2;
            }
        };
    }

    private ScoreFunction.ExactScoreFunction euclideanScoreFunctionFrom(VectorFloat<?> query, LVQPackedVectors packedVectors) {
        /*
         * The query vector (full resolution) will be compared to LVQ quantized vectors that were first de-meaned.
         * Rather than re-adding the global mean to all quantized vectors, we can shift the query vector the same amount.
         * This will result in the same squared L2 distances with less work.
         */
        var shiftedQuery = VectorUtil.sub(query, globalMean);
        return new ScoreFunction.ExactScoreFunction() {
            @Override
            public VectorFloat<?> similarityTo(int[] nodes) {
                var results = vts.createFloatVector(nodes.length);
                var nodeCount = nodes.length;
                for (int i = 0; i < nodeCount; i++) {
                    var node = nodes[i];
                    var vector = packedVectors.getPackedVector(node);
                    var lvqDist = VectorUtil.lvqSquareL2Distance(shiftedQuery, vector);
                    results.set(i, 1 / (1 + lvqDist));
                }
                return results;
            }

            @Override
            public float similarityTo(int node2) {
                var vector = packedVectors.getPackedVector(node2);
                var lvqDist = VectorUtil.lvqSquareL2Distance(shiftedQuery, vector);
                return 1 / (1 + lvqDist);
            }
        };
    }

    private ScoreFunction.ExactScoreFunction cosineScoreFunctionFrom(VectorFloat<?> query, LVQPackedVectors packedVectors) {
        return new ScoreFunction.ExactScoreFunction() {
            @Override
            public VectorFloat<?> similarityTo(int[] nodes) {
                var results = vts.createFloatVector(nodes.length);
                var nodeCount = nodes.length;
                for (int i = 0; i < nodeCount; i++) {
                    var node = nodes[i];
                    var vector = packedVectors.getPackedVector(node);
                    var lvqCosine = VectorUtil.lvqCosine(query, vector, globalMean);
                    results.set(i, (1 + lvqCosine) / 2);
                }
                return results;
            }

            @Override
            public float similarityTo(int node2) {
                var vector = packedVectors.getPackedVector(node2);
                var lvqCosine = VectorUtil.lvqCosine(query, vector, globalMean);
                return (1 + lvqCosine) / 2;
            }
        };
    }

    public ScoreFunction.ExactScoreFunction scoreFunctionFrom(VectorFloat<?> query, VectorSimilarityFunction similarityFunction, LVQPackedVectors packedVectors) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return dotProductScoreFunctionFrom(query, packedVectors);
            case EUCLIDEAN:
                return euclideanScoreFunctionFrom(query, packedVectors);
            case COSINE:
                return cosineScoreFunctionFrom(query, packedVectors);
            default:
                throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
        }
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        throw new UnsupportedOperationException("LVQ does not produce a compressed vectors implementation");
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

    /**
     * A LVQ-quantized vector. This has not been packed for storage, but it can be written in a packed format.
     * Packed vectors are necessary to use VectorUtil LVQ similarity functions.
     */
    public static class QuantizedVector {
        private final ByteSequence<?> bytes;
        private final float bias;
        private final float scale;

        public QuantizedVector(ByteSequence<?> bytes, float bias, float scale) {
            this.bytes = bytes;
            this.bias = bias;
            this.scale = scale;
        }

        // safely write a byte from the encodedVector or 0 if out of bounds
        private void writeByteSafely(DataOutput out, ByteSequence<?> encodedVector, int index) throws IOException {
            if (index < encodedVector.length()) {
                out.writeByte(encodedVector.get(index));
            } else {
                out.writeByte(0);
            }
        }

        public void writePacked(DataOutput out) throws IOException {
            // write the min and max
            out.writeFloat(bias);
            out.writeFloat(scale);

            // write the main part of the encoded vector that is evenly divisible into 64-byte blocks
            int mainBlockCount = bytes.length() / 64;
            int i;
            for (i = 0; i < mainBlockCount; i++) {
                var startIndex = i * 64;
                for (int j = startIndex; j < startIndex + 16; j++) {
                    out.writeByte(bytes.get(j));
                    out.writeByte(bytes.get(j + 16));
                    out.writeByte(bytes.get(j + 32));
                    out.writeByte(bytes.get(j + 48));
                }
            }

            // write the "tail" bytes from the last partial block
            var startIndex = i * 64;
            if (startIndex < bytes.length()) {
                var endIndex = Math.min(startIndex + 16, bytes.length());
                int j = startIndex;
                for (; j < endIndex; j++) {
                    writeByteSafely(out, bytes, j);
                    writeByteSafely(out, bytes, j + 16);
                    writeByteSafely(out, bytes,  j + 32);
                    writeByteSafely(out, bytes, j + 48);
                }
                for (; j < startIndex + 16; j++) {
                    out.writeInt(0);
                }
            }
        }
    }

    /**
     * A Turbo LVQ vector that has been packed into a byte sequence.
     * 64 bytes are packed into a block using LVQ-8, optimizing for AVX-512.
     * Helper methods are provided to get the quantized and dequantized values.
     */
    public static class PackedVector {
        public final ByteSequence<?> bytes;
        public final float bias;
        public final float scale;

        public PackedVector(ByteSequence<?> bytes, float bias, float scale) {
            this.bytes = bytes;
            this.bias = bias;
            this.scale = scale;
        }

        /**
         * Get the quantized value at the given index. This is the raw unsigned byte value. Note that the index reflects
         * the original index of the vector before it was packed.
         * @param index the index to get the quantized value for
         * @return the quantized value
         */
        public int getQuantized(int index) {
            var blockId = index / 64; // 64 bytes per block
            var inBlockId = index % 64; // switch to an in-block index
            var laneId = inBlockId % 16; // 4 bytes per lane
            var laneOffset = inBlockId / 16; // 16 lanes per block
            var packedIndex = blockId * 64 + laneId * 4 + laneOffset;
            return Byte.toUnsignedInt(bytes.get(packedIndex));
        }

        /**
         * Get the dequantized value at the given index. Note that the index reflects the original index of the vector
         * before it was packed.
         * @param index the index to get the dequantized value for
         * @return the dequantized value
         */
        public float getDequantized(int index) {
            return (getQuantized(index) * scale) + bias;
        }

        public PackedVector copy() {
            return new PackedVector(bytes.copy(), bias, scale);
        }
    }
}
