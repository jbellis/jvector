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

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TestNVQDecoding {
    @Test
    void fastLanesDecodeMatchesScalarBitsForEveryByteAndPart() {
        var support = new PanamaVectorUtilSupport();
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        byte[] bytes = new byte[ByteVector.SPECIES_PREFERRED.length()];
        for (int code = 0; code < 256; code++) {
            // Across code iterations, every coordinate exercises every unsigned byte.
            for (int i = 0; i < bytes.length; i++) bytes[i] = (byte) (code + i);
            var packed = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, bytes, 0);
            for (float inverseAlpha : new float[] {0.2f, 0.5f, 1f, 2f, 10f}) {
                for (float midpoint : new float[] {-0.25f, 0f, 0.25f}) {
                    for (int part = 0; part < 4; part++) {
                        float[] actual = support.nvqDequantize8bit(
                                packed, inverseAlpha, midpoint, 0.003f, 0.05f, part).toArray();
                        for (int lane = 0; lane < lanes; lane++) {
                            float value = Math.fma(Byte.toUnsignedInt(bytes[4 * lane + part]), 0.003f, 0.05f);
                            float z = value / (1f - value);
                            int bits = Float.floatToIntBits(z);
                            float exponent = ((bits & 0x7f800000) >>> 23) - 128;
                            float mantissa = Float.intBitsToFloat((bits & 0x007fffff) + 0x3f800000);
                            float expected = Math.fma(mantissa + exponent, inverseAlpha, midpoint);
                            assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(actual[lane]),
                                    "code=" + code + " part=" + part + " lane=" + lane);
                        }
                    }
                }
            }
        }
    }
}
