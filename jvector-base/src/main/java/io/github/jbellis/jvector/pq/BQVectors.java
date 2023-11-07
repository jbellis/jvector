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
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

public class BQVectors implements CompressedVectors {
    private final BinaryQuantization bq;
    private final long[][] compressedVectors;

    public BQVectors(BinaryQuantization bq, long[][] compressedVectors) {
        this.bq = bq;
        this.compressedVectors = compressedVectors;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        // BQ centering data
        bq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        if (compressedVectors.length <= 0) {
            return;
        }
        out.writeInt(compressedVectors[0].length);
        for (var v : compressedVectors) {
            for (int i = 0; i < v.length; i++) {
                out.writeLong(v[i]);
            }
        }
    }

    public static BQVectors load(RandomAccessReader in, int offset) throws IOException {
        in.seek(offset);

        // BQ
        var bq = BinaryQuantization.load(in);

        // check validity of compressed vectors header
        int size = in.readInt();
        if (size < 0) {
            throw new IOException("Invalid compressed vector count " + size);
        }
        var compressedVectors = new long[size][];
        if (size == 0) {
            return new BQVectors(bq, compressedVectors);
        }
        int compressedLength = in.readInt();
        if (compressedLength < 0) {
            throw new IOException("Invalid compressed vector dimension " + compressedLength);
        }

        // read the compressed vectors
        for (int i = 0; i < size; i++)
        {
            long[] vector = new long[compressedLength];
            in.readFully(vector);
            compressedVectors[i] = vector;
        }

        return new BQVectors(bq, compressedVectors);
    }

    @Override
    public NeighborSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        var qBQ = bq.encode(q);
        return node2 -> {
            var vBQ = compressedVectors[node2];
            return 1 - (float) VectorUtil.hammingDistance(qBQ, vBQ) / q.length;
        };
    }

    public long[] get(int i) {
        return compressedVectors[i];
    }

    @Override
    public int getOriginalSize() {
        return bq.getOriginalDimension() * Float.BYTES;
    }

    @Override
    public int getCompressedSize() {
        return compressedVectors[0].length * Long.BYTES;
    }

    @Override
    public long ramBytesUsed() {
        return compressedVectors.length * RamUsageEstimator.sizeOf(compressedVectors[0]);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BQVectors bqVectors = (BQVectors) o;
        return Objects.equals(bq, bqVectors.bq) && Arrays.deepEquals(compressedVectors, bqVectors.compressedVectors);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(bq);
        result = 31 * result + Arrays.deepHashCode(compressedVectors);
        return result;
    }
}
