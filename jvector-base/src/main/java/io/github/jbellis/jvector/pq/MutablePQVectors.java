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

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import static java.lang.Math.max;

public class MutablePQVectors extends PQVectors implements MutableCompressedVectors<VectorFloat<?>> {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    /**
     * Construct a mutable PQVectors instance with the given ProductQuantization and maximum number of vectors that will be
     * stored in this instance. The vectors are split into chunks to avoid exceeding the maximum array size.
     * @param pq the ProductQuantization to use
     * @param maximumVectorCount the maximum number of vectors that will be stored in this instance
     */
    public MutablePQVectors(ProductQuantization pq, int maximumVectorCount) {
        super(pq);
        this.vectorCount = 0;

        // Calculate if we need to split into multiple chunks
        int compressedDimension = pq.compressedVectorSize();
        long totalSize = (long) maximumVectorCount * compressedDimension;
        this.vectorsPerChunk = totalSize <= MAX_CHUNK_SIZE ? maximumVectorCount : MAX_CHUNK_SIZE / compressedDimension;

        int fullSizeChunks = maximumVectorCount / vectorsPerChunk;
        int totalChunks = maximumVectorCount % vectorsPerChunk == 0 ? fullSizeChunks : fullSizeChunks + 1;
        ByteSequence<?>[] chunks = new ByteSequence<?>[totalChunks];
        int chunkBytes = vectorsPerChunk * compressedDimension;
        for (int i = 0; i < fullSizeChunks; i++)
            chunks[i] = vectorTypeSupport.createByteSequence(chunkBytes);

        // Last chunk might be smaller
        if (totalChunks > fullSizeChunks) {
            int remainingVectors = maximumVectorCount % vectorsPerChunk;
            chunks[fullSizeChunks] = vectorTypeSupport.createByteSequence(remainingVectors * compressedDimension);
        }

        this.compressedDataChunks = chunks;
    }

    @Override
    public void encodeAndSet(int ordinal, VectorFloat<?> vector) {
        vectorCount = max(vectorCount, ordinal + 1);
        pq.encodeTo(vector, get(ordinal));
    }

    @Override
    public void setZero(int ordinal) {
        vectorCount = max(vectorCount, ordinal + 1);
        get(ordinal).zero();
    }
}
