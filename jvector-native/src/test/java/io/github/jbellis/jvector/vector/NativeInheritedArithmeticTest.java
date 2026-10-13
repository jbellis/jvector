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
package io.github.jbellis.jvector.vector;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class NativeInheritedArithmeticTest {
    private static final int[] LENGTHS = {0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 256, 768, 1536, 3072};

    @Test
    void inheritedSumAcceptsNativeStorage() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] values = new float[length];
            float expected = 0;
            for (int i = 0; i < length; i++) {
                // Exactly representable values and sums make the oracle independent of reduction order.
                values[i] = ((i * 17) % 127 - 63) / 256f;
                expected += values[i];
            }
            assertEquals(expected, support.sum(new MemorySegmentVectorFloat(values)), "length=" + length);
        }
    }

    @Test
    void inheritedScalePreservesSpecialValuesAndTails() {
        var support = new NativeVectorUtilSupport();
        float[] special = {0f, -0f, Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MIN_VALUE, -1f, 0.12345f};
        for (int length : LENGTHS) {
            for (float factor : new float[]{0f, -0f, 0.25f, -2f, Float.NaN, Float.POSITIVE_INFINITY}) {
                float[] values = new float[length];
                for (int i = 0; i < length; i++) values[i] = special[i % special.length];
                var actual = new MemorySegmentVectorFloat(values.clone());
                support.scale(actual, factor);
                for (int i = 0; i < length; i++) {
                    assertEquals(Float.floatToIntBits(values[i] * factor), Float.floatToIntBits(actual.get(i)),
                            "length=" + length + " lane=" + i);
                }
            }
        }
    }

    @Test
    void inheritedMinAndScalarSubtractionAcceptNativeStorage() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] values = new float[length];
            float expectedMin = Float.MAX_VALUE;
            for (int i = 0; i < length; i++) {
                values[i] = (i % 19 - 9) / 16f;
                expectedMin = Math.min(expectedMin, values[i]);
            }
            var vector = new MemorySegmentVectorFloat(values.clone());
            assertEquals(expectedMin, support.min(vector));
            var result = support.sub(vector, 0.25f);
            assertEquals(length, result.length());
            for (int i = 0; i < length; i++) {
                assertEquals(values[i] - 0.25f, result.get(i));
                assertEquals(values[i], vector.get(i));
            }
        }
    }

    @Test
    void inheritedSubtractionAcceptsNativeInputsAndPreservesOffsets() {
        var support = new NativeVectorUtilSupport();
        for (int length : LENGTHS) {
            float[] a = new float[length + 9], b = new float[length + 11];
            for (int i = 0; i < a.length; i++) a[i] = i * 0.125f - 4;
            for (int i = 0; i < b.length; i++) b[i] = i * -0.0625f + 2;
            var av = new MemorySegmentVectorFloat(a.clone());
            var bv = new MemorySegmentVectorFloat(b.clone());
            var result = support.sub(av, 3, bv, 7, length);
            assertEquals(length, result.length());
            for (int i = 0; i < length; i++) assertEquals(a[i + 3] - b[i + 7], result.get(i));
            // Inputs remain unchanged, including when both arguments reference the same storage.
            var aliased = support.sub(av, 1, av, 3, length);
            for (int i = 0; i < length; i++) assertEquals(a[i + 1] - a[i + 3], aliased.get(i));
            for (int i = 0; i < a.length; i++) assertEquals(a[i], av.get(i));
            for (int i = 0; i < b.length; i++) assertEquals(b[i], bv.get(i));
        }
    }
}
