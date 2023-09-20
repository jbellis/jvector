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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.graph.GraphIndexTestCase;
import io.github.jbellis.jvector.vector.types.ArrayVectorProvider;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

public class TestVectorizationProvider extends RandomizedTest {
    static boolean hasSimd = VectorizationProvider.vectorModulePresentAndReadable();

    @Test
    public void testDotProductFloat() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorTypeSupport aTypes = new ArrayVectorProvider();

        VectorizationProvider b = VectorizationProvider.getInstance();
        VectorTypeSupport bTypes = VectorizationProvider.getInstance().getVectorTypeSupport();

        for (int i = 0; i < 1000; i++) {
            float[] v1 = GraphIndexTestCase.randomVector(getRandom(), 1021); //prime numbers
            float[] v2 = GraphIndexTestCase.randomVector(getRandom(), 1021); //prime numbers

            VectorFloat<?> v1a = aTypes.createFloatType(v1);
            VectorFloat<?> v2a = aTypes.createFloatType(v2);

            VectorFloat<?> v1b = bTypes.createFloatType(v1);
            VectorFloat<?> v2b = bTypes.createFloatType(v2);

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a,v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b), 0.00001f);
        }
    }

    @Test
    public void testDotProductByte() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorTypeSupport aTypes = new ArrayVectorProvider();

        VectorizationProvider b = VectorizationProvider.getInstance();
        VectorTypeSupport bTypes = VectorizationProvider.getInstance().getVectorTypeSupport();

        for (int i = 0; i < 1000; i++) {
            byte[] v1 = GraphIndexTestCase.randomVector8(getRandom(), 1021); //prime numbers
            byte[] v2 = GraphIndexTestCase.randomVector8(getRandom(), 1021); //prime numbers

            VectorByte<?> v1a = aTypes.createByteType(v1);
            VectorByte<?> v2a = aTypes.createByteType(v2);

            VectorByte<?> v1b = bTypes.createByteType(v1);
            VectorByte<?> v2b = bTypes.createByteType(v2);

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b));
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a,v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b));
        }
    }
}
