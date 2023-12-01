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

import io.github.jbellis.jvector.disk.Io;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.min;

/**
 * Product Quantization for float vectors.  Supports arbitrary source and target dimensionality;
 * in particular, the source does not need to be evenly divisible by the target.
 * <p>
 * Codebook cluster count is fixed at 256.
 */
public class ProductQuantization implements VectorCompressor<byte[]> {
    static final int CLUSTERS = 256; // number of clusters per subspace = one byte's worth
    static final int K_MEANS_ITERATIONS = 6;
    static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

    final float[][][] codebooks;
    private final int M; // codebooks.length, redundantly reproduced for convenience
    final int originalDimension;
    private final float[] globalCentroid;
    final int[][] subvectorSizesAndOffsets;

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public static ProductQuantization compute(RandomAccessVectorValues<float[]> ravv, int M, boolean globallyCenter) {
        return compute(ravv, M, globallyCenter, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public static ProductQuantization compute(
            RandomAccessVectorValues<float[]> ravv,
            int M,
            boolean globallyCenter,
            ForkJoinPool simdExecutor,
            ForkJoinPool parallelExecutor) {
        // limit the number of vectors we train on
        var P = min(1.0f, MAX_PQ_TRAINING_SET_SIZE / (float) ravv.size());
        var ravvCopy = ravv.isValueShared() ? PoolingSupport.newThreadBased(ravv::copy) : PoolingSupport.newNoPooling(ravv);
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectors = parallelExecutor.submit(() -> IntStream.range(0, ravv.size()).parallel()
                .filter(i -> ThreadLocalRandom.current().nextFloat() < P)
                .mapToObj(targetOrd -> {
                    try (var pooledRavv = ravvCopy.get()) {
                        var localRavv = pooledRavv.get();
                        float[] v = localRavv.vectorValue(targetOrd);
                        return localRavv.isValueShared() ? Arrays.copyOf(v, v.length) : v;
                    }
                })
                .collect(Collectors.toList()))
                .join();

        // subtract the centroid from each training vector
        float[] globalCentroid;
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            List<float[]> finalVectors = vectors;
            vectors = simdExecutor.submit(() -> finalVectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.toList())).join();
        } else {
            globalCentroid = null;
        }

        // derive the codebooks
        var codebooks = createCodebooks(vectors, M, subvectorSizesAndOffsets, simdExecutor);
        return new ProductQuantization(codebooks, globalCentroid);
    }

    ProductQuantization(float[][][] codebooks, float[] globalCentroid)
    {
        this.codebooks = codebooks;
        this.globalCentroid = globalCentroid;
        this.M = codebooks.length;
        this.subvectorSizesAndOffsets = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
            int size = codebooks[i][0].length;
            this.subvectorSizesAndOffsets[i] = new int[]{size, offset};
            offset += size;
        }
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new PQVectors(this, (byte[][]) compressedVectors);
    }

    /**
     * Encodes the given vectors in parallel using the PQ codebooks.
     */
    @Override
    public byte[][] encodeAll(List<float[]> vectors, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() ->vectors.stream().parallel().map(this::encode).toArray(byte[][]::new)).join();
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     *
     * @return one byte per subspace
     */
    @Override
    public byte[] encode(float[] vector) {
        if (globalCentroid != null) {
            vector = VectorUtil.sub(vector, globalCentroid);
        }

        float[] finalVector = vector;
        byte[] encoded = new byte[M];
        for (int m = 0; m < M; m++) {
            encoded[m] = (byte) closetCentroidIndex(getSubVector(finalVector, m, subvectorSizesAndOffsets), codebooks[m]);
        }
        return encoded;
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector.
     */
    public void decode(byte[] encoded, float[] target) {
        decodeCentered(encoded, target);

        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            VectorUtil.addInPlace(target, globalCentroid);
        }
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector, relative to the global centroid.
     */
    void decodeCentered(byte[] encoded, float[] target) {
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = codebooks[m][centroidIndex];
            System.arraycopy(centroidSubvector, 0, target, subvectorSizesAndOffsets[m][1], subvectorSizesAndOffsets[m][0]);
        }
    }

    /**
     * @return how many bytes we are compressing to
     */
    public int getSubspaceCount() {
        return M;
    }

    // for testing
    static void printCodebooks(List<List<float[]>> codebooks) {
        List<List<String>> strings = codebooks.stream()
                .map(L -> L.stream()
                        .map(ProductQuantization::arraySummary)
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
        System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream()
                .map(L -> "[" + String.join(", ", L) + "]")
                .collect(Collectors.toList())));
    }
    private static String arraySummary(float[] a) {
        List<String> b = new ArrayList<>();
        for (int i = 0; i < min(4, a.length); i++) {
            b.add(String.valueOf(a[i]));
        }
        if (a.length > 4) {
            b.set(3, "... (" + a.length + ")");
        }
        return "[" + String.join(", ", b) + "]";
    }

    static float[][][] createCodebooks(List<float[]> vectors, int M, int[][] subvectorSizeAndOffset, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    float[][] subvectors = vectors.stream().parallel()
                            .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
                            .toArray(float[][]::new);
                    var clusterer = new KMeansPlusPlusClusterer(subvectors, CLUSTERS, VectorUtil::squareDistance);
                    return clusterer.cluster(K_MEANS_ITERATIONS);
                })
                .toArray(float[][][]::new))
                .join();
    }
    
    static int closetCentroidIndex(float[] subvector, float[][] codebook) {
        int index = 0;
        float minDist = Integer.MAX_VALUE;
        for (int i = 0; i < codebook.length; i++) {
            float dist = VectorUtil.squareDistance(subvector, codebook[i]);
            if (dist < minDist) {
                minDist = dist;
                index = i;
            }
        }
        return index;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     */
    static float[] getSubVector(float[] vector, int m, int[][] subvectorSizeAndOffset) {
        float[] subvector = new float[subvectorSizeAndOffset[m][0]];
        System.arraycopy(vector, subvectorSizeAndOffset[m][1], subvector, 0, subvectorSizeAndOffset[m][0]);
        return subvector;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
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

    public void write(DataOutput out) throws IOException
    {
        if (globalCentroid == null) {
            out.writeInt(0);
        } else {
            out.writeInt(globalCentroid.length);
            Io.writeFloats(out, globalCentroid);
        }

        out.writeInt(M);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        assert M == subvectorSizesAndOffsets.length;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }

        assert codebooks.length == M;
        assert codebooks[0].length == CLUSTERS;
        out.writeInt(codebooks[0].length);
        for (var codebook : codebooks) {
            for (var centroid : codebook) {
                Io.writeFloats(out, centroid);
            }
        }
    }

    public static ProductQuantization load(RandomAccessReader in) throws IOException {
        int globalCentroidLength = in.readInt();
        float[] globalCentroid = null;
        if (globalCentroidLength > 0) {
            globalCentroid = new float[globalCentroidLength];
            in.readFully(globalCentroid);
        }

        int M = in.readInt();
        int[][] subvectorSizes = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            offset += size;
            subvectorSizes[i][1] = offset;
        }

        int clusters = in.readInt();
        float[][][] codebooks = new float[M][][];
        for (int m = 0; m < M; m++) {
            float[][] codebook = new float[clusters][];
            for (int i = 0; i < clusters; i++) {
                int n = subvectorSizes[m][0];
                float[] centroid = new float[n];
                in.readFully(centroid);
                codebook[i] = centroid;
            }
            codebooks[m] = codebook;
        }

        return new ProductQuantization(codebooks, globalCentroid);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ProductQuantization that = (ProductQuantization) o;
        return M == that.M
               && originalDimension == that.originalDimension
               && Arrays.deepEquals(codebooks, that.codebooks)
               && Arrays.equals(globalCentroid, that.globalCentroid)
               && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(M, originalDimension);
        result = 31 * result + Arrays.deepHashCode(codebooks);
        result = 31 * result + Arrays.hashCode(globalCentroid);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    public float[] getCenter() {
        return globalCentroid;
    }

    public long memorySize() {
        long size = 0;
        for (float[][] codebook : codebooks) {
            for (float[] floats : codebook) {
                size += RamUsageEstimator.sizeOf(floats);
            }
        }

        return size;
    }

    @Override
    public String toString() {
        return String.format("ProductQuantization(%s)", M);
    }
}
