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
import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

public class CompressedVectors
{
    final ProductQuantization pq;
    private final List<byte[]> compressedVectors;
    private final ThreadLocal<float[][]> partialSums; // for dot product, euclidean, and cosine
    private final ThreadLocal<float[][]> partialMagnitudes; // for cosine

    public CompressedVectors(ProductQuantization pq, List<byte[]> compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
        this.partialSums = ThreadLocal.withInitial(() -> initFloatFragments(pq));
        this.partialMagnitudes = ThreadLocal.withInitial(() -> initFloatFragments(pq));
    }

    private static float[][] initFloatFragments(ProductQuantization pq) {
        float[][] a = new float[pq.getSubspaceCount()][];
        for (int i = 0; i < a.length; i++) {
            a[i] = new float[ProductQuantization.CLUSTERS];
        }
        return a;
    }

    public void write(DataOutput out) throws IOException
    {
        // pq codebooks
        pq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.size());
        out.writeInt(pq.getSubspaceCount());
        for (var v : compressedVectors) {
            out.write(v);
        }
    }

    public static CompressedVectors load(RandomAccessReader in, long offset) throws IOException
    {
        in.seek(offset);

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        var compressedVectors = new ArrayList<byte[]>(size);
        int compressedDimension = in.readInt();
        for (int i = 0; i < size; i++)
        {
            byte[] vector = new byte[compressedDimension];
            in.readFully(vector);
            compressedVectors.add(vector);
        }

        return new CompressedVectors(pq, compressedVectors);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CompressedVectors that = (CompressedVectors) o;
        if (!Objects.equals(pq, that.pq)) return false;
        if (compressedVectors.size() != that.compressedVectors.size()) return false;
        return IntStream.range(0, compressedVectors.size()).allMatch((i) -> {
            return Arrays.equals(compressedVectors.get(i), that.compressedVectors.get(i));
        });
    }

    @Override
    public int hashCode() {
        return Objects.hash(pq, compressedVectors);
    }

    public NeighborSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new CompressedDecoder.DotProductDecoder(this, q);
            case EUCLIDEAN:
                return new CompressedDecoder.EuclideanDecoder(this, q);
            case COSINE:
                return new CompressedDecoder.CosineDecoder(this, q);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    public byte[] get(int ordinal) {
        return compressedVectors.get(ordinal);
    }

    float[][] reusablePartialSums() {
        return partialSums.get();
    }

    float[][] reusablePartialMagnitudes() {
        return partialMagnitudes.get();
    }
}
