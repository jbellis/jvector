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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KMeansInitializationTest {
    @Test
    void initializationMatchesMainWithinReductionPrecision() {
        var nativeTypes = new MemorySegmentVectorProvider();
        var javaProvider = new PanamaVectorizationProvider();
        var javaTypes = javaProvider.getVectorTypeSupport();
        var main = javaProvider.getVectorUtilSupport();
        var support = new NativeVectorUtilSupport();
        var random = new Random(8719);
        for (int count : new int[]{1, 7, 17, 63, 256, 4099, 32768}) {
            VectorFloat<?>[] points = new VectorFloat<?>[count];
            VectorFloat<?>[] reference = new VectorFloat<?>[count];
            for (int i = 0; i < count; i++) {
                float[] values = new float[8];
                for (int j = 0; j < 8; j++) {
                    values[j] = (random.nextFloat() - .5f) * 12;
                    if (i % 4 == 3) values[j] = Float.intBitsToFloat(random.nextInt());
                }
                if (i == 0) Arrays.fill(values, -0.0f);
                if (i == 1) Arrays.fill(values, Float.POSITIVE_INFINITY);
                if (i == 2) Arrays.fill(values, Float.NaN);
                points[i] = nativeTypes.createFloatVector(values);
                reference[i] = javaTypes.createFloatVector(values);
            }
            var expectedMin = javaTypes.createFloatVector(count);
            var actualMin = nativeTypes.createFloatVector(count);
            var expectedDistances = javaTypes.createFloatVector(count);
            var actualDistances = nativeTypes.createFloatVector(count);
            for (int pass = 0; pass < 8; pass++) {
                for (int i = 0; i < count; i++) {
                    float value = pass == 0 ? Float.MAX_VALUE : switch (i % 8) {
                        case 0 -> -0.0f; case 1 -> 0.0f; case 2 -> Float.NaN;
                        case 3 -> Float.POSITIVE_INFINITY; case 4 -> Float.NEGATIVE_INFINITY;
                        default -> Float.MAX_VALUE;
                    };
                    expectedMin.set(i, value);
                    actualMin.set(i, value);
                }
                float[] values = new float[8];
                for (int j = 0; j < 8; j++) values[j] = (random.nextFloat() - .5f) * 12;
                if (pass == 7) Arrays.fill(values, Float.POSITIVE_INFINITY);
                var centroid = nativeTypes.createFloatVector(values);
                var referenceCentroid = javaTypes.createFloatVector(values);
                for (int i = 0; i < count; i++)
                    expectedDistances.set(i, main.squareDistance(reference[i], referenceCentroid));
                main.minInPlace(expectedMin, expectedDistances);
                assertTrue(support.tryUpdateKMeansDistances(points, centroid, actualMin, actualDistances));
                for (int i = 0; i < count; i++) {
                    sameDistance(expectedDistances.get(i), actualDistances.get(i));
                    sameDistance(distanceOracle(reference[i], referenceCentroid), actualDistances.get(i));
                    sameDistance(expectedMin.get(i), actualMin.get(i));
                    float originalMin = pass == 0 ? Float.MAX_VALUE : switch (i % 8) {
                        case 0 -> -0.0f; case 1 -> 0.0f; case 2 -> Float.NaN;
                        case 3 -> Float.POSITIVE_INFINITY; case 4 -> Float.NEGATIVE_INFINITY;
                        default -> Float.MAX_VALUE;
                    };
                    sameBits(Math.min(originalMin, actualDistances.get(i)), actualMin.get(i));
                }
            }
        }
    }

    @Test
    void unsupportedPathsLeaveOutputsAlone() {
        var nativeTypes = new MemorySegmentVectorProvider();
        var support = new NativeVectorUtilSupport();
        var point = nativeTypes.createFloatVector(new float[4]);
        var minimum = nativeTypes.createFloatVector(new float[]{17});
        var scratch = nativeTypes.createFloatVector(new float[]{23});
        assertFalse(support.tryUpdateKMeansDistances(new VectorFloat<?>[]{point}, point, minimum, scratch));
        assertEquals(17, minimum.get(0));
        assertEquals(23, scratch.get(0));
        var javaSupport = new PanamaVectorizationProvider().getVectorUtilSupport();
        assertFalse(javaSupport.tryUpdateKMeansDistances(new VectorFloat<?>[]{point}, point, minimum, scratch));
        assertEquals(17, minimum.get(0));
        assertEquals(23, scratch.get(0));
    }

    private static float distanceOracle(VectorFloat<?> point, VectorFloat<?> centroid) {
        double sum = 0;
        for (int i = 0; i < 8; i++) {
            float difference = point.get(i) - centroid.get(i);
            float square = difference * difference;
            sum += square;
        }
        return (float) sum;
    }

    private static void sameDistance(float expected, float actual) {
        // FloatVector ADD reductions may use any order, even within one JVM.
        // Eight nonnegative squared terms have at most seven rounded additions;
        // allow eight ULPs against both the provider and a double-sum oracle.
        if (Float.isFinite(expected) && expected != 0.0f) {
            assertTrue(Float.isFinite(actual));
            assertTrue(Math.abs((double) expected - actual) <= 8.0 * Math.ulp(expected),
                       "expected=" + expected + " actual=" + actual);
        } else {
            sameBits(expected, actual);
        }
    }

    private static void sameBits(float expected, float actual) {
        if (Float.isNaN(expected)) assertTrue(Float.isNaN(actual));
        else assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }
}
