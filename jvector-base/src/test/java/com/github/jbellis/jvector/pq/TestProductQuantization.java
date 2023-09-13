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

package com.github.jbellis.jvector.pq;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.github.jbellis.jvector.example.util.SimpleMappedReader;
import org.junit.Test;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    @Test
    public void testSaveLoad() throws IOException {
        // Generate a PQ for random 2D vectors
        var vectors = IntStream.range(0, 512).mapToObj(i -> new float[]{getRandom().nextFloat(), getRandom().nextFloat()}).collect(Collectors.toList());
        var pq = new ProductQuantization(vectors, 1, false);

        // Write the pq object
        File tempFile = File.createTempFile("pqtest", ".bin");
        tempFile.deleteOnExit();
        try (var out = new DataOutputStream(new FileOutputStream(tempFile))) {
            pq.write(out);
        }

        // Read the pq object
        try (var in = new SimpleMappedReader(tempFile.getAbsolutePath())) {
            var pq2 = ProductQuantization.load(in);
            assertEquals(pq, pq2);
        }
    }
}
