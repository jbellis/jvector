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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class PQVectors implements CompressedVectors, ProductQuantization.EncodedSource {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    final ProductQuantization pq;
    private final List<ByteSequence<?>> compressedVectors;

    /**
     * Initialize the PQVectors with an initial List of vectors.  This list may be
     * mutated, but caller is responsible for thread safety issues when doing so.
     */
    public PQVectors(ProductQuantization pq, List<ByteSequence<?>> compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
    }

    public PQVectors(ProductQuantization pq, ByteSequence<?>[] compressedVectors)
    {
        this(pq, List.of(compressedVectors));
    }

    @Override
    public int count() {
        return compressedVectors.size();
    }

    @Override
    public void write(DataOutput out, int version) throws IOException
    {
        // pq codebooks
        pq.write(out, version);

        // compressed vectors
        out.writeInt(compressedVectors.size());
        out.writeInt(pq.getSubspaceCount());
        for (var v : compressedVectors) {
            vectorTypeSupport.writeByteSequence(out, v);
        }
    }

    public static PQVectors load(RandomAccessReader in) throws IOException {
        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        List<ByteSequence<?>> compressedVectors = new ArrayList<>(size);

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        for (int i = 0; i < size; i++)
        {
            ByteSequence<?> vector = vectorTypeSupport.readByteSequence(in, compressedDimension);
            compressedVectors.add(vector);
        }

        return new PQVectors(pq, compressedVectors);
    }

    public static PQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        return load(in);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        return Objects.equals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pq, compressedVectors);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new PQDecoder.DotProductDecoder(this, q);
            case EUCLIDEAN:
                return new PQDecoder.EuclideanDecoder(this, q);
            case COSINE:
                return new PQDecoder.CosineDecoder(this, q);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return pq.scoreFunctionFor(q, similarityFunction, this);
    }

    public ByteSequence<?> get(int ordinal) {
        return compressedVectors.get(ordinal);
    }

    public ProductQuantization getProductQuantization() {
        return pq;
    }

    VectorFloat<?> reusablePartialSums() {
        return pq.reusablePartialSums();
    }

    AtomicReference<VectorFloat<?>> partialSquaredMagnitudes() {
        return pq.partialSquaredMagnitudes();
    }

    @Override
    public int getOriginalSize() {
        return pq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return pq.compressedVectorSize();
    }

    @Override
    public ProductQuantization getCompressor() {
        return pq;
    }

    @Override
    public long ramBytesUsed() {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AH_BYTES = RamUsageEstimator.NUM_BYTES_ARRAY_HEADER;

        long codebooksSize = pq.ramBytesUsed();
        long listSize = (long) REF_BYTES * (1 + compressedVectors.size());
        long dataSize = (long) (OH_BYTES + AH_BYTES + pq.compressedVectorSize()) * compressedVectors.size();
        return codebooksSize + listSize + dataSize;
    }

    @Override
    public String toString() {
        return "PQVectors{" +
                "pq=" + pq +
                ", count=" + compressedVectors.size() +
                '}';
    }
}
