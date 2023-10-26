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

import io.github.jbellis.jvector.util.PhysicalCoreExecutor;

import java.util.List;

/**
 * A Product Quantization implementation for float vectors.
 */
public class BinaryQuantization {
    public long[][] encodeAll(List<float[]> vectors) {
        return PhysicalCoreExecutor.instance.submit(() -> vectors.stream().parallel().map(this::encode).toArray(long[][]::new));
    }

    /**
     * Encodes the input vector
     *
     * @return one bit per original f32
     */
    public long[] encode(float[] vector) {
        int M = (int) Math.ceil(vector.length / 64.0);
        long[] encoded = new long[M];
        for (int i = 0; i < M; i++) {
            long bits = 0;
            for (int j = 0; j < 64; j++) {
                int idx = i * 64 + j;
                if (idx >= vector.length) {
                    break;
                }
                if (vector[idx] > 0) {
                    bits |= 1L << j;
                }
            }
            encoded[i] = bits;
        }
        return encoded;
    }
}
