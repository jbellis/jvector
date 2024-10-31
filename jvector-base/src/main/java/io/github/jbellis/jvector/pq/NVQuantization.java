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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;


/**
 * Non-uniform Vector Quantization for float vectors.
 */
public class NVQuantization implements VectorCompressor<NVQuantization.QuantizedSubVector>, Accountable {
    private static final int MAGIC = 0x75EC4012; // JVECTOR, with some imagination

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public final VectorFloat<?> globalMean;
    public final int originalDimension;
    public final int[][] subvectorSizesAndOffsets;

    private NVQuantization(int[][] subvectorSizesAndOffsets, VectorFloat<?> globalMean) {
        this.globalMean = globalMean;
        this.subvectorSizesAndOffsets = subvectorSizesAndOffsets;
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();

        if (globalMean.length() != originalDimension) {
            var msg = String.format("Global mean length %d does not match vector dimensionality %d", globalMean.length(), originalDimension);
            throw new IllegalArgumentException(msg);
        }
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param nSubVectors number of subspaces
     */
    public static NVQuantization compute(RandomAccessVectorValues ravv, int nSubVectors) {
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), nSubVectors);

        var ravvCopy = ravv.threadLocalSupplier().get();
        // convert ravvCopy to list
        var dim = ravvCopy.getVector(0).length();
        var globalMean = vectorTypeSupport.createFloatVector(dim);
        globalMean.zero();
        for (int i = 0; i < ravvCopy.size(); i++) {
            VectorUtil.addInPlace(globalMean, ravvCopy.getVector(i));
        }
        return new NVQuantization(subvectorSizesAndOffsets, globalMean);
    }


    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new NVQVectors(this, (QuantizedSubVector[]) compressedVectors);
    }

    /**
     * Encodes the given vectors in parallel using the PQ codebooks.
     */
    @Override
    public QuantizedSubVector[] encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> IntStream.range(0, ravv.size())
                        .parallel()
                        .mapToObj(i -> encode(ravv.getVector(i)))
                        .toArray(QuantizedSubVector[]::new))
                .join();
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     * @return one byte per subspace
     */
    @Override
    public QuantizedSubVector encode(VectorFloat<?> vector) {
        // TODO right now, this is applied to the full vector, think how to apply to to subvectors

        vector = VectorUtil.sub(vector, globalMean);

        return new QuantizedSubVector(vector, 8);
    }

    /**
     * Extracts the m-th subvector from a single vector.
     */
    static VectorFloat<?> getSubVector(VectorFloat<?> vector, int m, int[][] subvectorSizeAndOffset) {
        VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(subvectorSizeAndOffset[m][0]);
        subvector.copyFrom(vector, subvectorSizeAndOffset[m][1], 0, subvectorSizeAndOffset[m][0]);
        return subvector;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    @VisibleForTesting
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
        if (M > dimensions) {
            throw new IllegalArgumentException("Number of subspaces must be less than or equal to the vector dimension");
        }
        int[][] sizes = new int[M][];
        int baseSize = dimensions / M;
        int remainder = dimensions % M;
        // distribute the remainder among the subvectors
        int offset = 0;
        for (int i = 0; i < M; i++) {
            int size = baseSize + (i < remainder ? 1 : 0);
            sizes[i] = new int[]{size, offset};
            offset += size;
        }
        return sizes;
    }

    public void write(DataOutput out, int version) throws IOException
    {
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported serialization version " + version);
        }

        if (version >= 3) {
            out.writeInt(MAGIC);
            out.writeInt(version);
        }

        out.writeInt(globalMean.length());
        vectorTypeSupport.writeFloatVector(out, globalMean);

        out.writeInt(subvectorSizesAndOffsets.length);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }
    }

    @Override
    public int compressorSize() {
        int size = 0;
        size += Integer.BYTES; // MAGIC
        size += Integer.BYTES; // STORAGE_VERSION
        size += Integer.BYTES; // globalCentroidLength
        size += Float.BYTES * globalMean.length();
        return size;
    }

    public static NVQuantization load(RandomAccessReader in) throws IOException {
        int maybeMagic = in.readInt();
        int version;
        int globalMeanLength;
        if (maybeMagic != MAGIC) {
            // JVector 1+2 format, no magic or version, starts straight off with the centroid length
            version = 0;
            globalMeanLength = maybeMagic;
        } else {
            version = in.readInt();
            globalMeanLength = in.readInt();
        }

        VectorFloat<?> globalMean = null;
        if (globalMeanLength > 0) {
            globalMean = vectorTypeSupport.readFloatVector(in, globalMeanLength);
        }

        int nSubVectors = in.readInt();
        int[][] subvectorSizes = new int[nSubVectors][];
        int offset = 0;
        for (int i = 0; i < nSubVectors; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            subvectorSizes[i][1] = offset;
            offset += size;
        }

        return new NVQuantization(subvectorSizes, globalMean);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NVQuantization that = (NVQuantization) o;
        return originalDimension == that.originalDimension
                && Objects.equals(globalMean, that.globalMean)
                && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(originalDimension);
        result = 31 * result + Objects.hashCode(globalMean);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    @Override
    public int compressedVectorSize() {
        return QuantizedSubVector.compressedVectorSize(globalMean.length());
    }

    @Override
    public long ramBytesUsed() {
        return globalMean.ramBytesUsed();
    }

    @Override
    public String toString() {
        return String.format("NVQuantization(sub-vectors=%d)", subvectorSizesAndOffsets.length);
    }

    /**
     * A NuVeQ vector. This has not been packed for storage, but it can be written in a packed format.
     * Packed vectors are necessary to use VectorUtil similarity functions.
     */
    public static class QuantizedSubVector {
        private final ByteSequence<?> bytes;
        int nBitsPerDimension;
        private final float kumaraswamyBias;
        private final float kumaraswamyScale;
        private final float kumaraswamyA;
        private final float kumaraswamyB;

        public static int compressedVectorSize(int nDims) {
            int roundedMeanLength = nDims % 64 == 0 ? nDims : (nDims / 64 + 1) * 64;
            return roundedMeanLength + 24* Float.BYTES;
        }

        public QuantizedSubVector(VectorFloat<?> vector, int nBitsPerDimension) {
            var u = VectorUtil.max(vector);
            var l = VectorUtil.min(vector);

            VectorUtil.subInPlace(vector, l);
            VectorUtil.scale(vector, u = l);

            // TODO do the Kumaraswamy training here
            float a = 0, b = 0;

            var quantized = uniformQuantize(vector, nBitsPerDimension);
            // TODO do the Kumaraswamy quantization here

            this.nBitsPerDimension = nBitsPerDimension;
            this.bytes = quantized;
            this.kumaraswamyBias = l;
            this.kumaraswamyScale = (u - l) / 255;
            this.kumaraswamyA = a;
            this.kumaraswamyB = b;
        }

        private QuantizedSubVector(ByteSequence<?> bytes, int nBitsPerDimension, float kumaraswamyBias,
                                   float kumaraswamyScale, float kumaraswamyA, float kumaraswamyB) {
            this.nBitsPerDimension = nBitsPerDimension;
            this.bytes = bytes;
            this.kumaraswamyBias = kumaraswamyBias;
            this.kumaraswamyScale = kumaraswamyScale;
            this.kumaraswamyA = kumaraswamyA;
            this.kumaraswamyB = kumaraswamyB;
        }

        // In-place quantization
        private ByteSequence<?> uniformQuantize(VectorFloat<?> vector, int nBits) {
            // TODO adjust this code depending on nBits
            var quantized = vectorTypeSupport.createByteSequence(vector.length());
            float constant = (float) (Math.pow(2, nBits) - 1);
            VectorUtil.scale(vector, constant);
            for (int d = 0; d < vector.length(); d++) {
                int quantizedValue = Math.round(vector.get(d));

                // Ensure the quantized value is within the 0 to 255 range
                if (quantizedValue < 0) quantizedValue = 0;
                if (quantizedValue > 255) quantizedValue = 255;
                quantized.set(d, (byte) quantizedValue);
            }
            return quantized;
        }

        private VectorFloat<?> uniformDequantize(ByteSequence<?> bytes, int nBits) {
            // TODO adjust this code depending on nBits
            var vector = vectorTypeSupport.createFloatVector(bytes.length());
            for (int d = 0; d < bytes.length(); d++) {
                vector.set(d, bytes.get(d));
            }
            float constant = (float) (Math.pow(2, nBits) - 1);
            VectorUtil.scale(vector, 1.f / constant);
            return vector;
        }

        // In-place application of the CDF of the Kumaraswamy distribution
        private void forwardKumaraswamy(VectorFloat<?> x, float a, float b) {
            // Compute 1 - (1 - v ** a) ** b
            VectorUtil.constantMinusExponentiatedVector(x, 1, a); // 1 - v ** a
            VectorUtil.constantMinusExponentiatedVector(x, 1, b); // 1 - v ** b
        }

        // In-place application of the inverse CDF of the Kumaraswamy distribution
        private void inverseKumaraswamy(VectorFloat<?> y, float a, float b) {
            // Compute (1 - (1 - y) ** (1 / b)) ** (1 / a)
            VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / b); // 1 - v ** (1 / a)
            VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / a); // 1 - v ** (1 / b)
        }

        // safely write a byte from the encodedVector or 0 if out of bounds
        private void writeByteSafely(DataOutput out, ByteSequence<?> encodedVector, int index) throws IOException {
            if (index < encodedVector.length()) {
                out.writeByte(encodedVector.get(index));
            } else {
                out.writeByte(0);
            }
        }

        public void write(DataOutput out) throws IOException {
            // write the min and max
            out.writeInt(nBitsPerDimension);
            out.writeFloat(kumaraswamyBias);
            out.writeFloat(kumaraswamyScale);
            out.writeFloat(kumaraswamyA);
            out.writeFloat(kumaraswamyB);
            out.writeInt(bytes.length());

            vectorTypeSupport.writeByteSequence(out, bytes);
        }

        public static QuantizedSubVector load(RandomAccessReader in) throws IOException {
            int nBitsPerDimension = in.readInt();
            float kumaraswamyBias = in.readFloat();
            float kumaraswamyScale = in.readFloat();
            float kumaraswamyA = in.readFloat();
            float kumaraswamyB = in.readFloat();
            int compressedDimension = in.readInt();

            ByteSequence<?> bytes = vectorTypeSupport.readByteSequence(in, compressedDimension);

            return new QuantizedSubVector(bytes, nBitsPerDimension, kumaraswamyBias, kumaraswamyScale, kumaraswamyA, kumaraswamyB);
        }
    }
}
