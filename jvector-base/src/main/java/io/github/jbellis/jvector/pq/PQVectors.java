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
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.util.UnsafeUtils;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

public class PQVectors implements CompressedVectors {
    final ProductQuantization pq;
    private final byte[][] compressedVectors;
    private final ByteBuffer compressedBuffer;
    private final long cbOffHeapAddress;
    private final int vectorsSize;
    private final int vectorDim;
    private final ThreadLocal<float[]> partialSums; // for dot product, euclidean, and cosine
    private final ThreadLocal<float[]> partialMagnitudes; // for cosine

    public PQVectors(ProductQuantization pq, byte[][] compressedVectors) {
        this.vectorsSize = compressedVectors.length;
        this.vectorDim = compressedVectors[0].length;
        this.compressedBuffer = null;
        this.cbOffHeapAddress = -1;
        this.compressedVectors = compressedVectors;
        this.pq = pq;
        this.partialSums = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
        this.partialMagnitudes = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
    }

    public PQVectors(ProductQuantization pq, ByteBuffer compressedVectors, int vectorsSize) {
        this.vectorsSize = vectorsSize;
        this.vectorDim = compressedVectors.limit() / vectorsSize;
        this.compressedBuffer = compressedVectors;
        this.cbOffHeapAddress = UnsafeUtils.getDirectBufferAddress(this.compressedBuffer);
        this.compressedVectors = null;
        this.pq = pq;
        this.partialSums = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
        this.partialMagnitudes = ThreadLocal.withInitial(() -> new float[pq.getSubspaceCount() * ProductQuantization.CLUSTERS]);
    }

    @Override
    public void write(DataOutput out) throws IOException
    {
        // pq codebooks
        pq.write(out);

        // compressed vectors
        out.writeInt(vectorsSize);
        out.writeInt(pq.getSubspaceCount());

        if (compressedVectors != null) {
            for (var v : compressedVectors) {
                out.write(v);
            }
        } else {
            compressedBuffer.position(0);
            int chunkSize = 4 * 1024 * 1024;
            byte[] chunk = new byte[chunkSize];
            while (compressedBuffer.hasRemaining()) {
                int remaining = compressedBuffer.remaining();
                int readSize = Math.min(remaining, chunkSize);
                compressedBuffer.get(chunk, 0, readSize);
                out.write(chunk, 0, readSize);
            }
        }
    }

    public static PQVectors load(RandomAccessReader in, long offset) throws IOException {
        return load(in, offset, false);
    }

    public static PQVectors load(RandomAccessReader in, long offset, boolean offHeap) throws IOException
    {
        in.seek(offset);

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }

        int compressedDimension = in.readInt();
        if (compressedDimension < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedDimension);
        }

        if (offHeap) {
            ByteBuffer cv = ByteBuffer.allocateDirect(compressedDimension * size).order(ByteOrder.LITTLE_ENDIAN);
            byte[] vector = new byte[compressedDimension];
            for (int i = 0; i < size; i++) {
                in.readFully(vector);
                cv.put(vector);
            }
            cv.flip();
            return new PQVectors(pq, cv, size);

        } else {
            var compressedVectors = new byte[size][];
            for (int i = 0; i < size; i++) {
                byte[] vector = new byte[compressedDimension];
                in.readFully(vector);
                compressedVectors[i] = vector;
            }
            return new PQVectors(pq, compressedVectors);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PQVectors that = (PQVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (compressedBuffer != null) {
            if (compressedBuffer.limit() != that.compressedBuffer.limit()) {
                return false;
            }
            for (int i = 0; i < compressedBuffer.limit(); i++) {
                if (compressedBuffer.get(i) != that.compressedBuffer.get(i)) {
                    return false;
                }
            }
            return true;
        } else {
            return Arrays.deepEquals(compressedVectors, that.compressedVectors);
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(pq, Arrays.deepHashCode(compressedVectors), compressedBuffer);
    }

    @Override
    public NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
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

    byte[] get(int ordinal) {
        if (compressedVectors != null) {
            return compressedVectors[ordinal];
        }
        byte[] bytes = new byte[vectorDim];
        UnsafeUtils.getBytes(this.cbOffHeapAddress + (long) ordinal * vectorDim, bytes, 0, vectorDim);
        return bytes;
    }

    float[] reusablePartialSums() {
        return partialSums.get();
    }

    float[] reusablePartialMagnitudes() {
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
    public long ramBytesUsed() {
        long codebooksSize = pq.memorySize();
        if (compressedVectors != null) {
            long compressedVectorSize = RamUsageEstimator.sizeOf(compressedVectors[0]);
            return codebooksSize + (compressedVectorSize * compressedVectors.length);
        }
        return codebooksSize + compressedBuffer.limit();
    }
}
