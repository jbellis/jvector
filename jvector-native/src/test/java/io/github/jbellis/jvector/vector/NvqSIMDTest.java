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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class NvqSIMDTest {
    @Test void shufflePreservesEveryBitAndTail() {
        VectorizationProvider.getInstance();
        Random random = new Random(6);
        for (int dim : new int[]{0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1536, 3072}) {
            float[] data = new float[dim];
            for (int i = 0; i < dim; i++) data[i] = Float.intBitsToFloat(random.nextInt());
            var vector = new MemorySegmentVectorFloat(data.clone());
            VectorUtil.nvqShuffleQueryInPlace8bit(vector);
            int step = 4 * jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length();
            for (int i = 0; i < dim; i++) {
                int block = i / step * step, p = i % step;
                int index = block + step <= dim ? block + (p % (step/4))*4 + p/(step/4) : i;
                assertEquals(Float.floatToRawIntBits(data[index]), Float.floatToRawIntBits(vector.get(i)), "shuffle " + dim + ":" + i);
            }
        }
    }

    @Test void meanPreservesSequentialOrderAndInputs() {
        VectorizationProvider.getInstance();
        Random random = new Random(60);
        for (int dim : new int[]{1, 3, 15, 16, 17, 63, 64, 65, 1536}) {
            List<VectorFloat<?>> values = new ArrayList<>();
            float[] expected = new float[dim];
            for (int n = 0; n < 257; n++) {
                float[] data = new float[dim];
                for (int i = 0; i < dim; i++) {data[i]=(float)random.nextGaussian();expected[i]+=data[i];}
                values.add(new MemorySegmentVectorFloat(data));
            }
            var ravv = new ListRandomAccessVectorValues(values, dim);
            var result = new MemorySegmentVectorFloat(dim);
            assertTrue(NativeNVQArithmetic.sum(ravv, result));
            for (int i = 0; i < dim; i++) assertEquals(Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(result.get(i)));
        }
    }
}
