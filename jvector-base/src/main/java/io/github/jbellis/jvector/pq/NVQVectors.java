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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

public class NVQVectors implements CompressedVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    final NVQuantization nvq;
    final NVQuantization.QuantizedVector[] compressedVectors;

    /**
     * Initialize the PQVectors with an initial List of vectors.  This list may be
     * mutated, but caller is responsible for thread safety issues when doing so.
     */
    public NVQVectors(NVQuantization nvq, List<NVQuantization.QuantizedVector> compressedVectors) {
        this(nvq, compressedVectors.toArray(new NVQuantization.QuantizedVector[0]));
    }

    public NVQVectors(NVQuantization nvq, NVQuantization.QuantizedVector[] compressedVectors) {
        this.nvq = nvq;
        this.compressedVectors = compressedVectors;
    }

    @Override
    public int count() {
        return compressedVectors.length;
    }

    @Override
    public void write(DataOutput out, int version) throws IOException
    {
        // pq codebooks
        nvq.write(out, version);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        for (var v : compressedVectors) {
            v.write(out);
        }
    }

    public static NVQVectors load(RandomAccessReader in) throws IOException {
        // pq codebooks
        var nvq = NVQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        NVQuantization.QuantizedVector[] compressedVectors = new NVQuantization.QuantizedVector[size];

        for (int i = 0; i < size; i++) {
            compressedVectors[i] = NVQuantization.QuantizedVector.load(in);
        }

        return new NVQVectors(nvq, compressedVectors);
    }

    public static NVQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        return load(in);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NVQVectors that = (NVQVectors) o;
        if (!Objects.equals(nvq, that.nvq)) return false;
        return Objects.equals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        return Objects.hash(nvq, compressedVectors);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(query, similarityFunction);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return dotProductScoreFunctionFor(query);
            case EUCLIDEAN:
                return euclideanScoreFunctionFor(query);
            case COSINE:
                return cosineScoreFunctionFor(query);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    private ScoreFunction.ApproximateScoreFunction dotProductScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         * The dot product is calculated between the query and quantized sub-vectors as follows:
         *
         * <query, vector> \approx <query, scale * quantized + bias + globalMean>
         *                       = scale * <query, quantized> + bias <query, broadcast(1)> + <query, globalMean>
         *
         * where scale and bias are scalars.
         *
         * The following terms can be precomputed:
         *     queryGlobalBias = <query, globalMean>
         *     querySum = <query, broadcast(1)>
         */
        var queryGlobalBias = VectorUtil.dotProduct(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(query);

        var querySum = new float[querySubVectors.length];
        for (int i = 0; i < querySubVectors.length; i++) {
            querySum[i] = VectorUtil.sum(querySubVectors[i]);
            VectorUtil.nvqShuffleQueryInPlace(querySubVectors[i], this.nvq.bitsPerDimension);
        }

        return node2 -> {
            var vNVQ = compressedVectors[node2];
            float nvqDot = 0;
            for (int i = 0; i < querySubVectors.length; i++) {
                var subVec = vNVQ.subVectors[i];
                nvqDot += VectorUtil.nvqDotProduct(querySubVectors[i], subVec, querySum[i]);
            }
            // TODO This won't work without some kind of normalization.  Intend to scale [0, 1]
            return (1 + nvqDot + queryGlobalBias) / 2;
        };
    }

    private ScoreFunction.ApproximateScoreFunction euclideanScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         *
         * The squared L2 distance is calculated between the query and quantized sub-vectors as follows:
         *
         * |query - vector|^2 \approx |query - scale * quantized + bias + globalMean|^2
         *                          = |(query - globalMean) - scale * quantized + bias|^2
         *
         * where scale and bias are scalars.
         *
         * The following term can be precomputed:
         *     shiftedQuery = query - globalMean
         */
        var shiftedQuery = VectorUtil.sub(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(shiftedQuery);

        for (VectorFloat<?> querySubVector : querySubVectors) {
            VectorUtil.nvqShuffleQueryInPlace(querySubVector, this.nvq.bitsPerDimension);
        }

        return node2 -> {
            var vNVQ = compressedVectors[node2];
            float dist = 0;
            for (int i = 0; i < querySubVectors.length; i++) {
                dist += VectorUtil.nvqSquareL2Distance(querySubVectors[i], vNVQ.subVectors[i]);
            }

            return 1 / (1 + dist);
        };
    }

    private ScoreFunction.ApproximateScoreFunction cosineScoreFunctionFor(VectorFloat<?> query) {
        float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
        var querySubVectors = this.nvq.getSubVectors(query);
        var meanSubVectors = this.nvq.getSubVectors(this.nvq.globalMean);

        for (VectorFloat<?> querySubVector : querySubVectors) {
            VectorUtil.nvqShuffleQueryInPlace(querySubVector, this.nvq.bitsPerDimension);
        }

        return node2 -> {
            var vNVQ = compressedVectors[node2];
            float dotProduct = 0;
            float squaredNormalization = 0;
            for (int i = 0; i < querySubVectors.length; i++) {
                var partialCosSim = VectorUtil.nvqCosine(querySubVectors[i], vNVQ.subVectors[i], meanSubVectors[i]);
                dotProduct += partialCosSim[0];
                squaredNormalization += partialCosSim[1];
            }
            float cosine = (dotProduct / queryNorm) / (float) Math.sqrt(squaredNormalization);

            return (1 + cosine) / 2;
        };
    }

    public NVQuantization.QuantizedVector get(int ordinal) {
        return compressedVectors[ordinal];
    }

    public NVQuantization getNVQuantization() {
        return nvq;
    }

    @Override
    public int getOriginalSize() {
        return nvq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return nvq.compressedVectorSize();
    }

    @Override
    public NVQuantization getCompressor() {
        return nvq;
    }

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long codebooksSize = nvq.ramBytesUsed();
        long listSize = (long) REF_BYTES * (1 + compressedVectors.length);
        long dataSize = (long) (OH_BYTES + AH_BYTES + nvq.compressedVectorSize()) * compressedVectors.length;
        return codebooksSize + listSize + dataSize;
    }

    @Override
    public String toString() {
        return "NVQVectors{" +
                "NVQ=" + nvq +
                ", count=" + compressedVectors.length +
                '}';
    }
}
