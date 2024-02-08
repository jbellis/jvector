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

import io.github.jbellis.jvector.disk.Io;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.min;

/**
 * Binary Quantization of float vectors: each float is compressed to a single bit,
 * and similarity is computed with a simple Hamming distance.
 */
public class BinaryQuantization implements VectorCompressor<long[]> {
    private final float[] globalCentroid;

    public BinaryQuantization(float[] globalCentroid) {
        this.globalCentroid = globalCentroid;
    }

    public static BinaryQuantization compute(RandomAccessVectorValues<float[]> ravv) {
        return compute(ravv, ForkJoinPool.commonPool());
    }

    public static BinaryQuantization compute(RandomAccessVectorValues<float[]> ravv, ForkJoinPool parallelExecutor) {
        var vectors = ProductQuantization.extractTrainingVectors(ravv, parallelExecutor);
        float[] globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
        return new BinaryQuantization(globalCentroid);
    }

    @Override
    public CompressedVectors createCompressedVectors(Object[] compressedVectors) {
        return new BQVectors(this, (long[][]) compressedVectors);
    }

    @Override
    public long[][] encodeAll(List<float[]> vectors, ForkJoinPool simdExecutor) {
        return simdExecutor.submit(() -> vectors.stream().parallel().map(this::encode).toArray(long[][]::new)).join();
    }

    /**
     * Encodes the input vector
     *
     * @return one bit per original f32
     */
    @Override
    public long[] encode(float[] v) {
        var centered = VectorUtil.sub(v, globalCentroid);

        int M = (int) Math.ceil(centered.length / 64.0);
        long[] encoded = new long[M];
        for (int i = 0; i < M; i++) {
            long bits = 0;
            for (int j = 0; j < 64; j++) {
                int idx = i * 64 + j;
                if (idx >= centered.length) {
                    break;
                }
                if (centered[idx] > 0) {
                    bits |= 1L << j;
                }
            }
            encoded[i] = bits;
        }
        return encoded;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(globalCentroid.length);
        Io.writeFloats(out, globalCentroid);
    }

    public int getOriginalDimension() {
        return globalCentroid.length;
    }

    public static BinaryQuantization load(RandomAccessReader in) throws IOException {
        int length = in.readInt();
        var centroid = new float[length];
        in.readFully(centroid);
        return new BinaryQuantization(centroid);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BinaryQuantization that = (BinaryQuantization) o;
        return Arrays.equals(globalCentroid, that.globalCentroid);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(globalCentroid);
    }

    @Override
    public String toString() {
        return "BinaryQuantization";
    }
}
