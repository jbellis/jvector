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
import io.github.jbellis.jvector.optimization.LossFunction;
import io.github.jbellis.jvector.optimization.NESOptimizer;
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
public class NVQuantization implements VectorCompressor<NVQuantization.QuantizedVector>, Accountable {
    public enum BitsPerDimension {
        EIGHT {
            @Override
            public int getInt() {
                return 8;
            }

            @Override
            public ByteSequence<?> uniformQuantize(VectorFloat<?> vector) {
                // TODO there is an opportunity for vectorization here
                var quantized = vectorTypeSupport.createByteSequence(vector.length());
                int constant = (1 << getInt()) - 1;
                VectorUtil.scale(vector, (float) constant);
                for (int d = 0; d < vector.length(); d++) {
                    // Ensure the quantized value is within the 0 to constant range
                    int quantizedValue = Math.min(Math.max(0, Math.round(vector.get(d))), constant);
                    quantized.set(d, (byte) quantizedValue);
                }
                return quantized;
            }

            @Override
            public VectorFloat<?> uniformDequantize(ByteSequence<?> bytes, int originalDimensions) {
                // TODO there is an opportunity for vectorization here
                var vector = vectorTypeSupport.createFloatVector(bytes.length());
                for (int d = 0; d < bytes.length(); d++) {
                    vector.set(d, Byte.toUnsignedInt(bytes.get(d)));
                }
                int constant = (1 << getInt()) - 1;
                VectorUtil.scale(vector, 1.f / constant);
                return vector;
            }
        },
        FOUR {
            @Override
            public int getInt() {
                return 4;
            };

            @Override
            public ByteSequence<?> uniformQuantize(VectorFloat<?> vector) {
                // TODO there is an opportunity for vectorization here
                var quantized = vectorTypeSupport.createByteSequence((int) Math.ceil(vector.length() / 2.));
                int constant = (1 << getInt()) - 1;
                VectorUtil.scale(vector, (float) constant);
                for (int d = 0; d < vector.length(); d += 2) {
                    // Ensure the quantized value is within the 0 to constant range
                    int quantizedValue0 = Math.min(Math.max(0, Math.round(vector.get(d))), constant);
                    int quantizedValue1;
                    if (d + 1 < vector.length()) {
                        quantizedValue1 = Math.min(Math.max(0, Math.round(vector.get(d + 1))), constant);
                    } else {
                        quantizedValue1 = 0;
                    }

                    quantized.set(d / 2, (byte) ((quantizedValue1 << getInt()) + quantizedValue0));
                }

                return quantized;
            }

            @Override
            public VectorFloat<?> uniformDequantize(ByteSequence<?> bytes, int originalDimensions) {
                // TODO there is an opportunity for vectorization here
                var vector = vectorTypeSupport.createFloatVector(originalDimensions);
                int constant = (1 << getInt()) - 1;
                for (int d = 0; d < bytes.length(); d++) {
                    int quantizedValue = Byte.toUnsignedInt(bytes.get(d));
                    vector.set(2 * d, quantizedValue & constant);
                    if (2 * d + 1 < vector.length()) {
                        vector.set(2 * d + 1, quantizedValue >> getInt());
                    }
                }
                VectorUtil.scale(vector, 1.f / constant);
                return vector;
            }
        };

        public void write(DataOutput out) throws IOException {
            out.writeInt(getInt());
        }

        public abstract ByteSequence<?> uniformQuantize(VectorFloat<?> vector);

        public abstract VectorFloat<?> uniformDequantize(ByteSequence<?> bytes, int originalDimensions);

        public abstract int getInt();

        public static BitsPerDimension load(RandomAccessReader in) throws IOException {
            int nBitsPerDimension = in.readInt();
            switch (nBitsPerDimension) {
                case 4:
                    return BitsPerDimension.FOUR;
                case 8:
                    return BitsPerDimension.EIGHT;
                default:
                    throw new IllegalArgumentException("Unsupported BitsPerDimension " + nBitsPerDimension);
            }
        }
    }

    private static final int MAGIC = 0x75EC4012; // JVECTOR, with some imagination

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    public final BitsPerDimension bitsPerDimension;
    public final VectorFloat<?> globalMean;
    public final int originalDimension;
    public final int[][] subvectorSizesAndOffsets;

    private NVQuantization(int[][] subvectorSizesAndOffsets, VectorFloat<?> globalMean, BitsPerDimension bitsPerDimension) {
        this.bitsPerDimension = bitsPerDimension;
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
    public static NVQuantization compute(RandomAccessVectorValues ravv, int nSubVectors, BitsPerDimension bitsPerDimension) {
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), nSubVectors);

        var ravvCopy = ravv.threadLocalSupplier().get();
        // convert ravvCopy to list
        var dim = ravvCopy.getVector(0).length();
        var globalMean = vectorTypeSupport.createFloatVector(dim);
        globalMean.zero();
        for (int i = 0; i < ravvCopy.size(); i++) {
            VectorUtil.addInPlace(globalMean, ravvCopy.getVector(i));
        }
        return new NVQuantization(subvectorSizesAndOffsets, globalMean, bitsPerDimension);
    }


    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new NVQVectors(this, (QuantizedVector[]) compressedVectors);
    }

    /**
     * Encodes the given vectors in parallel using NVQ.
     */
    @Override
    public QuantizedVector[] encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> IntStream.range(0, ravv.size())
                        .parallel()
                        .mapToObj(i -> encode(ravv.getVector(i)))
                        .toArray(QuantizedVector[]::new))
                .join();
    }

    /**
     * Encodes the input vector using NVQ.
     * @return one subvector per subspace
     */
    @Override
    public QuantizedVector encode(VectorFloat<?> vector) {
        vector = VectorUtil.sub(vector, globalMean);
        return new QuantizedVector(getSubVectors(vector), bitsPerDimension);
    }

    /**
     * Creates an array of subvectors from a given vector
     */
    @VisibleForTesting
    public VectorFloat<?>[] getSubVectors(VectorFloat<?> vector) {
        VectorFloat<?>[] subvectors = new VectorFloat<?>[subvectorSizesAndOffsets.length];

        // Iterate through the subvectorSizesAndOffsets to create each subvector and copy slices into them
        for (int i = 0; i < subvectorSizesAndOffsets.length; i++) {
            int size = subvectorSizesAndOffsets[i][0];   // Size of the subvector
            int offset = subvectorSizesAndOffsets[i][1]; // Offset from the start of the input vector
            VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(size);
            subvector.copyFrom(vector, offset, 0, size);
            subvectors[i] = subvector;
        }
        return subvectors;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    @VisibleForTesting
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
        if (M > dimensions) {
            throw new IllegalArgumentException("Number of subspaces must be less than or equal to the vector dimension");
        }
        int[][] sizes = new int[M][2];
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

        bitsPerDimension.write(out);

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

        BitsPerDimension bitsPerDimension = BitsPerDimension.load(in);

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

        return new NVQuantization(subvectorSizes, globalMean, bitsPerDimension);
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
        int size = 0;
        for (int[] subvectorSizesAndOffset : subvectorSizesAndOffsets) {
            size += QuantizedSubVector.compressedVectorSize(subvectorSizesAndOffset[0], bitsPerDimension);
        }
        return size;
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
     * A NuVeQ vector.
     */
    public static class QuantizedVector {
        public final QuantizedSubVector[] subVectors;

        public QuantizedVector(VectorFloat<?>[] subVectors, BitsPerDimension bitsPerDimension) {
            this.subVectors = new QuantizedSubVector[subVectors.length];
            for (int i = 0; i < subVectors.length; i++) {
                this.subVectors[i] = new QuantizedSubVector(subVectors[i], bitsPerDimension);
            }
        }

        private QuantizedVector(QuantizedSubVector[] subVectors) {
            this.subVectors = subVectors;
        }

        public void write(DataOutput out) throws IOException {
            // write the min and max
            out.writeInt(subVectors.length);

            for (var sv : subVectors) {
                sv.write(out);
            }
        }

        public static QuantizedVector load(RandomAccessReader in) throws IOException {
            int length = in.readInt();
            var subVectors = new QuantizedSubVector[length];
            for (int i = 0; i < length; i++) {
                subVectors[i] = QuantizedSubVector.load(in);
            }

            return new QuantizedVector(subVectors);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuantizedVector that = (QuantizedVector) o;
            return Arrays.deepEquals(subVectors, that.subVectors);
        }
    }

    /**
     * A NuVeQ sub-vector.
     */
    public static class QuantizedSubVector {
        public final ByteSequence<?> bytes;
        public BitsPerDimension bitsPerDimension;
        public final float kumaraswamyA;
        public final float kumaraswamyB;
        public final float kumaraswamyBias;
        public final float kumaraswamyScale;
        public final int originalDimensions;

        public static int compressedVectorSize(int nDims, BitsPerDimension bitsPerDimension) {
            // Here we assume that an enum takes 4 bytes
            switch (bitsPerDimension) {
                case EIGHT: return nDims + 4 * Float.BYTES + 4;
                case FOUR: return (int) Math.ceil(nDims / 2.) + 4 * Float.BYTES + 4;
                default: return 0; // never realized
            }
        }

        public QuantizedSubVector(VectorFloat<?> vector, BitsPerDimension bitsPerDimension) {
            var u = VectorUtil.max(vector);
            var l = VectorUtil.min(vector);

            VectorUtil.subInPlace(vector, l);
            VectorUtil.scale(vector, 1.f / (u - l));

            //-----------------------------------------------------------------
            // Optimization to find the hyperparameters of the Kumaraswamy quantization
            var loss = new KumaraswamyQuantizationLossFunction(2, bitsPerDimension, vector);
            loss.setMinBounds(new double[]{1e-6, 1e-6});

            double[] initialSolution = {1, 1};
            var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

            var tolerance = 1e-6;
            xnes.setTol(tolerance);
            var sol = xnes.optimize(loss, initialSolution, 0.5);
            float a = (float) sol.x[0];
            float b = (float) sol.x[1];
            //-----------------------------------------------------------------

            forwardKumaraswamy(vector, a, b);
            var quantized = bitsPerDimension.uniformQuantize(vector);

            this.bitsPerDimension = bitsPerDimension;
            this.kumaraswamyBias = l;
            this.kumaraswamyScale = (u - l);
            this.kumaraswamyA = a;
            this.kumaraswamyB = b;
            this.bytes = quantized;
            this.originalDimensions = vector.length();
        }

        private QuantizedSubVector(ByteSequence<?> bytes, int originalDimensions, BitsPerDimension bitsPerDimension,
                                   float kumaraswamyBias, float kumaraswamyScale,
                                   float kumaraswamyA, float kumaraswamyB) {
            this.bitsPerDimension = bitsPerDimension;
            this.bytes = bytes;
            this.kumaraswamyBias = kumaraswamyBias;
            this.kumaraswamyScale = kumaraswamyScale;
            this.kumaraswamyA = kumaraswamyA;
            this.kumaraswamyB = kumaraswamyB;
            this.originalDimensions = originalDimensions;
        }

        // Does not apply the scale and bias, provided for vectorization purposes of similarity computations
        public VectorFloat<?> getDequantizedUnormalized() {
            var vector = bitsPerDimension.uniformDequantize(bytes, this.originalDimensions);
            inverseKumaraswamy(vector, kumaraswamyA, kumaraswamyB);
            return vector;
        }

        public VectorFloat<?> getDequantized() {
            var vector = bitsPerDimension.uniformDequantize(bytes, this.originalDimensions);
            inverseKumaraswamy(vector, kumaraswamyA, kumaraswamyB);
            VectorUtil.scale(vector, kumaraswamyScale);
            VectorUtil.addInPlace(vector, kumaraswamyBias);
            return vector;
        }

        public void write(DataOutput out) throws IOException {
            bitsPerDimension.write(out);
            out.writeFloat(kumaraswamyBias);
            out.writeFloat(kumaraswamyScale);
            out.writeFloat(kumaraswamyA);
            out.writeFloat(kumaraswamyB);
            out.writeInt(originalDimensions);
            out.writeInt(bytes.length());

            vectorTypeSupport.writeByteSequence(out, bytes);
        }

        public static QuantizedSubVector load(RandomAccessReader in) throws IOException {
            BitsPerDimension bitsPerDimension = BitsPerDimension.load(in);
            float kumaraswamyBias = in.readFloat();
            float kumaraswamyScale = in.readFloat();
            float kumaraswamyA = in.readFloat();
            float kumaraswamyB = in.readFloat();
            int originalDimensions = in.readInt();
            int compressedDimension = in.readInt();

            ByteSequence<?> bytes = vectorTypeSupport.readByteSequence(in, compressedDimension);

            return new QuantizedSubVector(bytes, originalDimensions, bitsPerDimension, kumaraswamyBias, kumaraswamyScale, kumaraswamyA, kumaraswamyB);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            QuantizedSubVector that = (QuantizedSubVector) o;
            return (kumaraswamyBias == that.kumaraswamyBias)
                    && (kumaraswamyScale == that.kumaraswamyScale)
                    && (kumaraswamyA == that.kumaraswamyA)
                    && (kumaraswamyB == that.kumaraswamyB)
                    && (bitsPerDimension == that.bitsPerDimension)
                    && bytes.equals(that.bytes);
        }
    }

    private static class KumaraswamyQuantizationLossFunction extends LossFunction {
        final private BitsPerDimension bitsPerDimension;
        final private VectorFloat<?> vectorOriginal;
        private VectorFloat<?> vectorCopy;

        public KumaraswamyQuantizationLossFunction(int nDims, BitsPerDimension bitsPerDimension, VectorFloat<?> vector) {
            super(nDims);
            this.bitsPerDimension = bitsPerDimension;
            vectorOriginal = vector;
            vectorCopy = vectorTypeSupport.createFloatVector(vectorOriginal.length());
        }

        public double compute(double[] x) {
            vectorCopy.copyFrom(vectorOriginal, 0, 0, vectorOriginal.length());
            forwardKumaraswamy(vectorCopy, (float) x[0], (float) x[1]);
            var bytes = bitsPerDimension.uniformQuantize(vectorCopy);
            vectorCopy = bitsPerDimension.uniformDequantize(bytes, vectorOriginal.length());
            inverseKumaraswamy(vectorCopy, (float) x[0], (float) x[1]);

            return -VectorUtil.squareL2Distance(vectorOriginal, vectorCopy);
        }
    }

    // In-place application of the CDF of the Kumaraswamy distribution
    private static void forwardKumaraswamy(VectorFloat<?> x, float a, float b) {
        // Compute 1 - (1 - v ** a) ** b
        VectorUtil.constantMinusExponentiatedVector(x, 1, a); // 1 - v ** a
        VectorUtil.constantMinusExponentiatedVector(x, 1, b); // 1 - v ** b
    }

    // In-place application of the inverse CDF of the Kumaraswamy distribution
    private static void inverseKumaraswamy(VectorFloat<?> y, float a, float b) {
        // Compute (1 - (1 - y) ** (1 / b)) ** (1 / a)
        VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / b); // 1 - v ** (1 / b)
        VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / a); // 1 - v ** (1 / a)
    }

}
