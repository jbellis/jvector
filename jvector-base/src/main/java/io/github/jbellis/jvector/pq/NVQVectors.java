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
    final NVQuantization.QuantizedSubVector[] compressedVectors;

    /**
     * Initialize the PQVectors with an initial List of vectors.  This list may be
     * mutated, but caller is responsible for thread safety issues when doing so.
     */
    public NVQVectors(NVQuantization nvq, List<NVQuantization.QuantizedSubVector> compressedVectors) {
        this(nvq, compressedVectors.toArray(new NVQuantization.QuantizedSubVector[0]));
    }

    public NVQVectors(NVQuantization nvq, NVQuantization.QuantizedSubVector[] compressedVectors) {
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
        NVQuantization.QuantizedSubVector[] compressedVectors = new NVQuantization.QuantizedSubVector[size];

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        for (int i = 0; i < size; i++) {
            compressedVectors[i] = NVQuantization.QuantizedSubVector.load(in);
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
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return implPrecomputedScoreFunctionFor(q, similarityFunction);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return implPrecomputedScoreFunctionFor(q, similarityFunction);
    }

    private ScoreFunction.ApproximateScoreFunction implPrecomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                //return new NVQDecoder.DotProductDecoder(this, q);
            case EUCLIDEAN:
                // return new NVQDecoder.EuclideanDecoder(this, q);
            case COSINE:
                // return new NVQDecoder.CosineDecoder(this, q);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    public NVQuantization.QuantizedSubVector get(int ordinal) {
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
