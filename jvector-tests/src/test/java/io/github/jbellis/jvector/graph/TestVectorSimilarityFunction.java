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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Assert;
import org.junit.Test;

import java.util.Collections;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class TestVectorSimilarityFunction extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    @Test
    public void testCompareMulti() {
        var random = getRandom();
        var dimension = random.nextInt(1535) + 1;
        var vectors = TestUtil.createRandomVectors(100, dimension);
        var q = TestUtil.randomVector(random, dimension);
        var length = random.nextInt(100) + 1;
        var indexes = IntStream.range(0, 100).boxed().collect(Collectors.toList());
        Collections.shuffle(indexes, random);
        var ids = indexes.subList(0, length).toArray(new Integer[0]);
        var results = vectorTypeSupport.createFloatVector(length);
        var packedVectors = vectorTypeSupport.createFloatVector(length * dimension);

        for (int i = 0; i < length; i++) {
            var v = vectors.get(ids[i]);
            packedVectors.copyFrom(v, 0, i * dimension, dimension);
        }

        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            vsf.compareMulti(q, packedVectors, results);
            for (int i = 0; i < length; i++) {
                Assert.assertEquals(vsf.compare(q, vectors.get(ids[i])), results.get(i), 0.01f);
            }
        }
    }
}
