package com.github.jbellis.jvector.pq;

import com.github.jbellis.jvector.disk.Io;
import com.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ProductQuantization {
    private static final int CLUSTERS = 256; // number of clusters per subspace = one byte's worth
    private static final int K_MEANS_ITERATIONS = 15; // VSTODO try 20 as well

    private final float[][][] codebooks;
    private final int M;
    private final int originalDimension;
    private final float[] globalCentroid;
    private final int[][] subvectorSizesAndOffsets;

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param vectors the points to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public ProductQuantization(List<float[]> vectors, int M, boolean globallyCenter) {
        this.M = M;
        originalDimension = vectors.get(0).length;
        subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(originalDimension, M);
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            vectors = vectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).toList();
        } else {
            globalCentroid = null;
        }
        codebooks = createCodebooks(vectors, M, subvectorSizesAndOffsets);
    }

    public ProductQuantization(float[][][] codebooks, float[] globalCentroid)
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

    /**
     * Encodes the given vectors in parallel using the PQ codebooks.
     */
    public List<byte[]> encodeAll(List<float[]> vectors) {
        return vectors.stream().parallel().map(this::encode).toList();
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     *
     * @return one byte per subspace
     */
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
     * Computes the dot product of the (approximate) original decoded vector with
     * another vector.
     *
     * If the PQ does not require centering, this method can compute the dot
     * product without materializing the decoded vector as a new float[], and will be
     * roughly 2x as fast as decode() + dot().
     */
    public float decodedDotProduct(byte[] encoded, float[] other) {
        if (globalCentroid != null) {
            float[] target = new float[originalDimension];
            decode(encoded, target);
            return VectorUtil.dotProduct(target, other);
        }

        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            int offset = subvectorSizesAndOffsets[m][1];
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = codebooks[m][centroidIndex];
            sum += VectorUtil.dotProduct(centroidSubvector, 0, other, offset, centroidSubvector.length);
        }

        return sum;
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector.
     */
    public void decode(byte[] encoded, float[] target) {
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = codebooks[m][centroidIndex];
            System.arraycopy(centroidSubvector, 0, target, subvectorSizesAndOffsets[m][1], subvectorSizesAndOffsets[m][0]);
        }

        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            VectorUtil.addInPlace(target, globalCentroid);
        }
    }

    /**
     * @return The dimension of the vectors being quantized.
     */
    public int getOriginalDimension() {
        return originalDimension;
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
                .toList();
        System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream()
                .map(L -> "[" + String.join(", ", L) + "]")
                .toList()));
    }
    private static String arraySummary(float[] a) {
        List<String> b = new ArrayList<>();
        for (int i = 0; i < Math.min(4, a.length); i++) {
            b.add(String.valueOf(a[i]));
        }
        if (a.length > 4) {
            b.set(3, "... (" + a.length + ")");
        }
        return "[" + String.join(", ", b) + "]";
    }

    static float[][][] createCodebooks(List<float[]> vectors, int M, int[][] subvectorSizeAndOffset) {
        return IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    float[][] subvectors = vectors.stream().parallel()
                            .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
                            .toArray(s -> new float[s][]);
                    var clusterer = new KMeansPlusPlusClusterer(subvectors, CLUSTERS, VectorUtil::squareDistance);
                    return clusterer.cluster(K_MEANS_ITERATIONS);
                })
                .toArray(s -> new float[s][][]);
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

    public static ProductQuantization load(DataInput in) throws IOException {
        int globalCentroidLength = in.readInt();
        float[] globalCentroid = null;
        if (globalCentroidLength > 0) {
            globalCentroid = Io.readFloats(in, globalCentroidLength);
        }

        int M = in.readInt();
        int[][] subvectorSizes = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
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
                float[] centroid = Io.readFloats(in, n);
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
        return M == that.M && originalDimension == that.originalDimension && Arrays.deepEquals(codebooks, that.codebooks) && Arrays.equals(globalCentroid, that.globalCentroid) && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(M, originalDimension);
        result = 31 * result + Arrays.deepHashCode(codebooks);
        result = 31 * result + Arrays.hashCode(globalCentroid);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }
}
