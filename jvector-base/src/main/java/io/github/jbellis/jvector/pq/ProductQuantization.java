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
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
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
public class ProductQuantization implements VectorCompressor<ByteSequence<?>> {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final int DEFAULT_CLUSTERS = 256; // number of clusters per subspace = one byte's worth
    static final int K_MEANS_ITERATIONS = 6;
    public static final int MAX_PQ_TRAINING_SET_SIZE = 128000;
    final VectorFloat<?>[] codebooks; // array of codebooks, where each codebook is a VectorFloat consisting of contiguous subvectors
    private final int M; // codebooks.length, redundantly reproduced for convenience
    private final int clusterCount; // codebooks[0].length, redundantly reproduced for convenience
    final int originalDimension;
    private final VectorFloat<?> globalCentroid;
    final int[][] subvectorSizesAndOffsets;

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param clusterCount number of clusters per subspace
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        return compute(ravv, M, clusterCount, globallyCenter, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization. Defaults to 256 clusters per subspace.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, boolean globallyCenter) {
        return compute(ravv, M, DEFAULT_CLUSTERS, globallyCenter);
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param clusterCount number of clusters per subspace
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public static ProductQuantization compute(
            RandomAccessVectorValues ravv,
            int M,
            int clusterCount,
            boolean globallyCenter,
            ForkJoinPool simdExecutor,
            ForkJoinPool parallelExecutor)
    {
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectors = extractTrainingVectors(ravv, parallelExecutor);

        // subtract the centroid from each training vector
        VectorFloat<?> globalCentroid;
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            List<VectorFloat<?>> finalVectors = vectors;
            vectors = simdExecutor.submit(() -> finalVectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        } else {
            globalCentroid = null;
        }

        // derive the codebooks
        var codebooks = createCodebooks(vectors, M, subvectorSizesAndOffsets, clusterCount, simdExecutor, parallelExecutor);
        return new ProductQuantization(codebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid);
    }

    static List<VectorFloat<?>> extractTrainingVectors(RandomAccessVectorValues ravv, ForkJoinPool parallelExecutor) {
        // limit the number of vectors we train on
        var P = min(1.0f, MAX_PQ_TRAINING_SET_SIZE / (float) ravv.size());
        var ravvCopy = ravv.threadLocalSupplier();
        return parallelExecutor.submit(() -> IntStream.range(0, ravv.size()).parallel()
                .filter(i -> ThreadLocalRandom.current().nextFloat() < P)
                .mapToObj(targetOrd -> {
                    var localRavv = ravvCopy.get();
                    VectorFloat<?> v = localRavv.vectorValue(targetOrd);
                    return localRavv.isValueShared() ? v.copy() : v;
                })
                .collect(Collectors.toList()))
                .join();
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv) {
        return refine(ravv, 1, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     *
     * @param lloydsRounds number of Lloyd's iterations to run against
     *                     the new data.  Suggested values are 1 or 2.
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv,
                                      int lloydsRounds,
                                      ForkJoinPool simdExecutor,
                                      ForkJoinPool parallelExecutor)
    {
        if (lloydsRounds < 0) {
            throw new IllegalArgumentException("lloydsRounds must be non-negative");
        }

        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectorsMutable = extractTrainingVectors(ravv, parallelExecutor);
        if (globalCentroid != null) {
            var vectors = vectorsMutable;
            vectorsMutable = simdExecutor.submit(() -> vectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        }
        var vectors = vectorsMutable; // "effectively final" to make the closure happy

        var refinedCodebooks = simdExecutor.submit(() -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizesAndOffsets, parallelExecutor);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, codebooks[m]);
            return clusterer.cluster(lloydsRounds);
        }).toArray(VectorFloat<?>[]::new)).join();

        return new ProductQuantization(refinedCodebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid);
    }

    ProductQuantization(VectorFloat<?>[] codebooks, int clusterCount, int[][] subvectorSizesAndOffsets, VectorFloat<?> globalCentroid) {
        this.codebooks = codebooks;
        this.globalCentroid = globalCentroid;
        this.M = codebooks.length;
        this.clusterCount = clusterCount;
        this.subvectorSizesAndOffsets = subvectorSizesAndOffsets;
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new PQVectors(this, (ByteSequence<?>[]) compressedVectors);
    }

    /**
     * Encodes the given vectors in parallel using the PQ codebooks.
     */
    @Override
    public ByteSequence<?>[] encodeAll(List<VectorFloat<?>> vectors, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> vectors.stream().parallel().map(this::encode).toArray(ByteSequence<?>[]::new)).join();
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     *
     * @return one byte per subspace
     */
    @Override
    public ByteSequence<?> encode(VectorFloat<?> vector) {
        if (globalCentroid != null) {
            vector = VectorUtil.sub(vector, globalCentroid);
        }

        VectorFloat<?> finalVector = vector;
        ByteSequence<?> encoded = vectorTypeSupport.createByteSequence(M);
        for (int m = 0; m < M; m++) {
            encoded.set(m, (byte) closestCentroidIndex(finalVector, subvectorSizesAndOffsets[m], codebooks[m]));
        }
        return encoded;
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector.
     */
    public void decode(ByteSequence<?> encoded, VectorFloat<?> target) {
        decodeCentered(encoded, target);

        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            VectorUtil.addInPlace(target, globalCentroid);
        }
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector, relative to the global centroid.
     */
    void decodeCentered(ByteSequence<?> encoded, VectorFloat<?> target) {
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
            target.copyFrom(codebooks[m], centroidIndex * subvectorSizesAndOffsets[m][0], subvectorSizesAndOffsets[m][1], subvectorSizesAndOffsets[m][0]);
        }
    }

    /**
     * @return how many bytes we are compressing to
     */
    public int getSubspaceCount() {
        return M;
    }


    /**
     * @return number of clusters per subspace
     */
    public int getClusterCount() {
        return clusterCount;
    }

    static VectorFloat<?>[] createCodebooks(List<VectorFloat<?>> vectors, int M, int[][] subvectorSizeAndOffset, int clusters, ForkJoinPool simdExecutor, ForkJoinPool parallelExecutor) {
        return simdExecutor.submit(() -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizeAndOffset, parallelExecutor);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, clusters);
            return clusterer.cluster(K_MEANS_ITERATIONS);
        }).toArray(VectorFloat<?>[]::new)).join();
    }

    /**
     * Extract VectorFloat subvectors corresponding to the m'th subspace.
     * This is NOT done in parallel (since the callers are themselves running in parallel).
     */
    private static VectorFloat<?>[] extractSubvectors(List<VectorFloat<?>> vectors, int m, int[][] subvectorSizeAndOffset, ForkJoinPool parallelExecutor) {
        return vectors.stream()
                .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
                .toArray(VectorFloat<?>[]::new);
    }
    
    static int closestCentroidIndex(VectorFloat<?> vector, int[] subvectorSizeAndOffset, VectorFloat<?> codebook) {
        int index = 0;
        float minDist = Integer.MAX_VALUE;
        // vectorFloat will have n subvectors, each of length subvectorSizeAndOffset[0]
        var clusterCount = codebook.length() / subvectorSizeAndOffset[0];
        for (int i = 0; i < clusterCount; i++) {
            float dist = VectorUtil.squareL2Distance(vector, subvectorSizeAndOffset[1], codebook, i * subvectorSizeAndOffset[0], subvectorSizeAndOffset[0]);
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
    static VectorFloat<?> getSubVector(VectorFloat<?> vector, int m, int[][] subvectorSizeAndOffset) {
        VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(subvectorSizeAndOffset[m][0]);
        subvector.copyFrom(vector, subvectorSizeAndOffset[m][1], 0, subvectorSizeAndOffset[m][0]);
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
            out.writeInt(globalCentroid.length());
            vectorTypeSupport.writeFloatVector(out, globalCentroid);
        }

        out.writeInt(M);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        assert M == subvectorSizesAndOffsets.length;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }

        assert codebooks.length == M;
        out.writeInt(clusterCount);
        for (int i = 0; i < M; i++) {
            var codebook = codebooks[i];
            assert codebook.length() == clusterCount * subvectorSizesAndOffsets[i][0];
            vectorTypeSupport.writeFloatVector(out, codebook);
        }
    }

    public static ProductQuantization load(RandomAccessReader in) throws IOException {
        int globalCentroidLength = in.readInt();
        VectorFloat<?> globalCentroid = null;
        if (globalCentroidLength > 0) {
            globalCentroid = vectorTypeSupport.readFloatVector(in, globalCentroidLength);
        }

        int M = in.readInt();
        int[][] subvectorSizes = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            subvectorSizes[i][1] = offset;
            offset += size;
        }

        int clusters = in.readInt();
        VectorFloat<?>[] codebooks = new VectorFloat<?>[M];
        for (int m = 0; m < M; m++) {
            VectorFloat<?> codebook = vectorTypeSupport.readFloatVector(in, clusters * subvectorSizes[m][0]);
            codebooks[m] = codebook;
        }

        return new ProductQuantization(codebooks, clusters, subvectorSizes, globalCentroid);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ProductQuantization that = (ProductQuantization) o;
        return M == that.M
                && originalDimension == that.originalDimension
                && Objects.equals(globalCentroid, that.globalCentroid)
                && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets)
                && Arrays.deepEquals(codebooks, that.codebooks);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(M, originalDimension);
        result = 31 * result + Arrays.deepHashCode(codebooks);
        result = 31 * result + Objects.hashCode(globalCentroid);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    public VectorFloat<?> getCenter() {
        return globalCentroid;
    }

    public long memorySize() {
        long size = 0;
        for (VectorFloat<?> codebook : codebooks) {
            size += codebook.ramBytesUsed();
        }

        return size;
    }

    @Override
    public String toString() {
        return String.format("ProductQuantization(%s,%s)", M, clusterCount);
    }
}
