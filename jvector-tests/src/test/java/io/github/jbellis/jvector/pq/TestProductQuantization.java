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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import org.junit.Test;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    @Test
    public void testPerfectReconstruction() {
        var vectors = IntStream.range(0,ProductQuantization.CLUSTERS).mapToObj(
                i -> new float[] {getRandom().nextInt(100000), getRandom().nextInt(100000), getRandom().nextInt(100000) })
                .collect(Collectors.toList());
        var ravv = new ListRandomAccessVectorValues(vectors, 3);
        var pq = ProductQuantization.compute(ravv, 2, false);
        var encoded = pq.encodeAll(vectors);
        var decodedScratch = new float[3];
        // if the number of vectors is equal to the number of clusters, we should perfectly reconstruct vectors
        for (int i = 0; i < vectors.size(); i++) {
            pq.decode(encoded[i], decodedScratch);
            assertArrayEquals(Arrays.toString(vectors.get(i)) + "!=" + Arrays.toString(decodedScratch), vectors.get(i), decodedScratch, 0);
        }
    }
}
