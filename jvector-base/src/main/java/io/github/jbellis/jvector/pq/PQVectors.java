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
import java.util.Arrays;
import java.util.Objects;

public class PQVectors implements CompressedVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    final ProductQuantization pq;
    private final ByteSequence<?>[] compressedVectors;
    private final ThreadLocal<VectorFloat<?>> partialSums; // for dot product, euclidean, and cosine
    private final ThreadLocal<VectorFloat<?>> partialMagnitudes; // for cosine

    public PQVectors(ProductQuantization pq, ByteSequence<?>[] compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
        this.partialSums = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(pq.getSubspaceCount() * pq.getClusterCount()));
        this.partialMagnitudes = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(pq.getSubspaceCount() * pq.getClusterCount()));
    }

    @Override
    public void write(DataOutput out) throws IOException
    {
        // pq codebooks
        pq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        out.writeInt(pq.getSubspaceCount());
        for (var v : compressedVectors) {
            vectorTypeSupport.writeByteSequence(out, v);
        }
    }

    public static PQVectors load(RandomAccessReader in, long offset) throws IOException
    {
        in.seek(offset);

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        var compressedVectors = new ByteSequence<?>[size];

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        for (int i = 0; i < size; i++)
        {
            ByteSequence<?> vector = vectorTypeSupport.readByteSequence(in, compressedDimension);
            compressedVectors[i] = vector;
        }

        return new PQVectors(pq, compressedVectors);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (compressedVectors.length != that.compressedVectors.length) return false;
        return Arrays.deepEquals(compressedVectors, that.compressedVectors);
    }

    @Override
    public int hashCode() {
        return Objects.hash(pq, Arrays.deepHashCode(compressedVectors));
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
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

    public ByteSequence<?> get(int ordinal) {
        return compressedVectors[ordinal];
    }

    public ProductQuantization getProductQuantization() {
        return pq;
    }

    VectorFloat<?> reusablePartialSums() {
        return partialSums.get();
    }

    VectorFloat<?> reusablePartialMagnitudes() {
        return partialMagnitudes.get();
    }

    @Override
    public int getOriginalSize() {
        return pq.originalDimension * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return pq.codebooks.length;
    }

    @Override
    public ProductQuantization getCompressor() {
        return pq;
    }

    @Override
    public long ramBytesUsed() {
        long codebooksSize = pq.memorySize();
        long compressedVectorSize = RamUsageEstimator.sizeOf(compressedVectors[0]);
        return codebooksSize + (compressedVectorSize * compressedVectors.length);
    }

    @Override
    public String toString() {
        return "PQVectors{" +
                "pq=" + pq +
                ", count=" + compressedVectors.length +
                '}';
    }
}
