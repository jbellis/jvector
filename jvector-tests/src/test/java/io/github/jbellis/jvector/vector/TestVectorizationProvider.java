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

import java.util.Random;

import io.github.jbellis.jvector.graph.GraphIndexTestCase;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;


public class TestVectorizationProvider {
    static boolean hasSimd = VectorizationProvider.vectorModulePresentAndReadable();
    static final Random r = new Random();
    @Test
    public void testSimilarityMetricsFloat() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorTypeSupport aTypes = new ArrayVectorProvider();

        VectorizationProvider b = VectorizationProvider.getInstance();
        VectorTypeSupport bTypes = VectorizationProvider.getInstance().getVectorTypeSupport();

        for (int i = 0; i < 1000; i++) {
            long seed = System.nanoTime();
            r.setSeed(seed);
            VectorFloat<?> v1a = GraphIndexTestCase.randomVector(aTypes, r, 2); //prime numbers so we benchmark tail
            VectorFloat<?> v2a = GraphIndexTestCase.randomVector(aTypes, r, 2); //prime numbers

            r.setSeed(seed);
            VectorFloat<?> v1b = GraphIndexTestCase.randomVector(bTypes, r, 2); //prime numbers
            VectorFloat<?> v2b = GraphIndexTestCase.randomVector(bTypes, r, 2); //prime numbers

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a,v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b), 0.00001f);
        }
    }

    @Test
    public void testSimilarityMetricsByte() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorTypeSupport aTypes = new ArrayVectorProvider();

        VectorizationProvider b = VectorizationProvider.getInstance();
        VectorTypeSupport bTypes = VectorizationProvider.getInstance().getVectorTypeSupport();

        for (int i = 0; i < 1000; i++) {
            long seed = System.nanoTime();
            r.setSeed(seed);
            VectorByte<?> v1a = GraphIndexTestCase.randomVector8(aTypes, r, 1021); //prime numbers so we benchmark tail
            VectorByte<?> v2a = GraphIndexTestCase.randomVector8(aTypes, r, 1021); //prime numbers

            r.setSeed(seed);
            VectorByte<?> v1b = GraphIndexTestCase.randomVector8(bTypes, r, 1021); //prime numbers
            VectorByte<?> v2b = GraphIndexTestCase.randomVector8(bTypes, r, 1021); //prime numbers

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1a,v2a), b.getVectorUtilSupport().dotProduct(v1b, v2b));
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1a,v2a), b.getVectorUtilSupport().cosine(v1b, v2b), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1a,v2a), b.getVectorUtilSupport().squareDistance(v1b, v2b));
        }
    }
}
