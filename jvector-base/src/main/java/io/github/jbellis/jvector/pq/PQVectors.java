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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

public class PQVectors implements CompressedVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final int MAX_CHUNK_SIZE = Integer.MAX_VALUE - 16; // standard Java array size limit with some headroom
    final ProductQuantization pq;
    private final ByteSequence<?>[] compressedDataChunks;
    private final int vectorCount;
    private final int vectorsPerChunk;
    private final boolean mutable;

    public PQVectors(ProductQuantization pq, int vectorCount)
    {
        this.pq = pq;
        this.mutable = true;
        this.vectorCount = vectorCount;

        // Calculate if we need to split into multiple chunks
        int compressedDimension = pq.compressedVectorSize();
        long totalSize = (long) vectorCount * compressedDimension;
        this.vectorsPerChunk = totalSize <= MAX_CHUNK_SIZE ? vectorCount : MAX_CHUNK_SIZE / compressedDimension;

        int numChunks = vectorCount / vectorsPerChunk;
        ByteSequence<?>[] chunks = new ByteSequence<?>[numChunks];
        int chunkSize = vectorsPerChunk * compressedDimension;
        for (int i = 0; i < numChunks - 1; i++)
            chunks[i] = vectorTypeSupport.createByteSequence(chunkSize);

        // Last chunk might be smaller
        int remainingVectors = vectorCount - (vectorsPerChunk * (numChunks - 1));
        chunks[numChunks - 1] = vectorTypeSupport.createByteSequence(remainingVectors * compressedDimension);

        compressedDataChunks = chunks;
    }

    public PQVectors(ProductQuantization pq, ByteSequence<?>[] compressedDataChunks, int vectorCount, int vectorsPerChunk)
    {
        this.pq = pq;
        this.mutable = false;
        this.compressedDataChunks = compressedDataChunks;
        this.vectorCount = vectorCount;
        this.vectorsPerChunk = vectorsPerChunk;
    }

    @Override
    public int count() {
        return vectorCount;
    }

    @Override
    public void write(DataOutput out, int version) throws IOException
    {
        // pq codebooks
        pq.write(out, version);

        // compressed vectors
        out.writeInt(vectorCount);
        out.writeInt(pq.getSubspaceCount());
        for (ByteSequence<?> chunk : compressedDataChunks) {
            vectorTypeSupport.writeByteSequence(out, chunk);
        }
    }

    public static PQVectors load(RandomAccessReader in) throws IOException {
        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int vectorCount = in.readInt();
        if (vectorCount < 0) {
            throw new IOException("Invalid compressed vector count " + vectorCount);
        }

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        // Calculate if we need to split into multiple chunks
        long totalSize = (long) vectorCount * compressedDimension;
        int vectorsPerChunk = totalSize <= MAX_CHUNK_SIZE ? vectorCount : MAX_CHUNK_SIZE / compressedDimension;

        int numChunks = vectorCount / vectorsPerChunk;
        ByteSequence<?>[] chunks = new ByteSequence<?>[numChunks];

        for (int i = 0; i < numChunks - 1; i++) {
            int chunkSize = vectorsPerChunk * compressedDimension;
            chunks[i] = vectorTypeSupport.readByteSequence(in, chunkSize);
        }

        // Last chunk might be smaller
        int remainingVectors = vectorCount - (vectorsPerChunk * (numChunks - 1));
        chunks[numChunks - 1] = vectorTypeSupport.readByteSequence(in, remainingVectors * compressedDimension);

        return new PQVectors(pq, chunks, vectorCount, vectorsPerChunk);
    }

    public static PQVectors load(RandomAccessReader in, long offset) throws IOException {
        in.seek(offset);
        return load(in);
    }

    /**
     * We consider two PQVectors equal when their PQs are equal and their compressed data is equal. We ignore the
     * chunking strategy in the comparison since this is an implementation detail.
     * @param o the object to check for equality
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (this.count() != that.count()) return false;
        // TODO how do we want to determine equality? With the current change, we are willing to write one
        // thing and materialize another. It seems like the real concern should be whether the compressedVectors have
        // the same data, not whether they are in a MemorySegment or a byte[] and not whether there is one chunk of many
        // vectors or many chunks of one vector. This technically goes against the implementation of each of the
        // ByteSequence#equals methods, which raises the question of whether this is valid. I primarily updated this
        // code to get testSaveLoadPQ to pass.
        for (int i = 0; i < this.count(); i++) {
            var thisNode = this.get(i);
            var thatNode = that.get(i);
            if (thisNode.length() != thatNode.length()) return false;
            for (int j = 0; j < thisNode.length(); j++) {
                if (thisNode.get(j) != thatNode.get(j)) return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        // We don't use the array structure in the hash code calculation because we allow for different chunking
        // strategies. Instead, we use the first entry in the first chunk to provide a stable hash code.
        return Objects.hash(pq, count(), count() > 0 ? get(0).get(0) : 0);
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
        VectorFloat<?> centeredQuery = pq.globalCentroid == null ? q : VectorUtil.sub(q, pq.globalCentroid);
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return (node2) -> {
                    var encoded = get(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float dp = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        dp += VectorUtil.dotProduct(pq.codebooks[m], centroidIndex * centroidLength, centeredQuery, centroidOffset, centroidLength);
                    }
                    // scale to [0, 1]
                    return (1 + dp) / 2;
                };
            case COSINE:
                float norm1 = VectorUtil.dotProduct(centeredQuery, centeredQuery);
                return (node2) -> {
                    var encoded = get(node2);
                    // compute the dot product of the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    float norm2 = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        var codebookOffset = centroidIndex * centroidLength;
                        sum += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, centeredQuery, centroidOffset, centroidLength);
                        norm2 += VectorUtil.dotProduct(pq.codebooks[m], codebookOffset, pq.codebooks[m], codebookOffset, centroidLength);
                    }
                    float cosine = sum / (float) Math.sqrt(norm1 * norm2);
                    // scale to [0, 1]
                    return (1 + cosine) / 2;
                };
            case EUCLIDEAN:
                return (node2) -> {
                    var encoded = get(node2);
                    // compute the euclidean distance between the query and the codebook centroids corresponding to the encoded points
                    float sum = 0;
                    for (int m = 0; m < pq.getSubspaceCount(); m++) {
                        int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
                        int centroidLength = pq.subvectorSizesAndOffsets[m][0];
                        int centroidOffset = pq.subvectorSizesAndOffsets[m][1];
                        sum += VectorUtil.squareL2Distance(pq.codebooks[m], centroidIndex * centroidLength, centeredQuery, centroidOffset, centroidLength);
                    }
                    // scale to [0, 1]
                    return 1 / (1 + sum);
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    public ByteSequence<?> get(int ordinal) {
        int chunkIndex = ordinal / vectorsPerChunk;
        int vectorIndexInChunk = ordinal % vectorsPerChunk;
        int start = vectorIndexInChunk * pq.getSubspaceCount();
        return compressedDataChunks[chunkIndex].slice(start, pq.getSubspaceCount());
    }

    /**
     * Encode the given vector and set it at the given ordinal. Done without unnecessary copying.
     * @param ordinal the ordinal to set
     * @param vector the vector to encode and set
     */
    public void encodeAndSet(int ordinal, VectorFloat<?> vector)
    {
        if (!mutable)
        {
            throw new UnsupportedOperationException("Cannot set values on an immutable PQVectors instance");
        }
        int chunkIndex = ordinal / vectorsPerChunk;
        int vectorIndexInChunk = ordinal % vectorsPerChunk;
        int start = vectorIndexInChunk * pq.getSubspaceCount();
        var slice = compressedDataChunks[chunkIndex].slice(start, pq.getSubspaceCount());
        pq.encodeTo(vector, slice);
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
        long chunksArraySize = OH_BYTES + AH_BYTES + (long) compressedDataChunks.length * REF_BYTES;
        long dataSize = 0;
        for (ByteSequence<?> chunk : compressedDataChunks) {
            dataSize += chunk.ramBytesUsed();
        }
        return codebooksSize + chunksArraySize + dataSize;
    }

    @Override
    public String toString() {
        return "PQVectors{" +
                "pq=" + pq +
                ", count=" + vectorCount +
                '}';
    }
}
