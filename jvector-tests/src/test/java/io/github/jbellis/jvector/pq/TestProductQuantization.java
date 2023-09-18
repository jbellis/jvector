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
import io.github.jbellis.jvector.disk.CompressedVectors;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import org.junit.Test;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestProductQuantization extends RandomizedTest {
    @Test
    public void testSaveLoad() throws Exception {
        // Generate a PQ for random 2D vectors
        var vectors = IntStream.range(0, 512).mapToObj(i -> new float[]{getRandom().nextFloat(), getRandom().nextFloat()}).collect(Collectors.toList());
        var pq = ProductQuantization.compute(new ListRandomAccessVectorValues(vectors, 2), 1, false);
        // Write the pq object
        File pqFile = File.createTempFile("pqtest", ".pq");
        pqFile.deleteOnExit();
        try (var out = new DataOutputStream(new FileOutputStream(pqFile))) {
            pq.write(out);
        }
        // Read the pq object
        try (var in = new SimpleMappedReader(pqFile.getAbsolutePath())) {
            var pq2 = ProductQuantization.load(in);
            assertEquals(pq, pq2);
        }

        // Compress the vectors and read and write those
        var compressed = pq.encodeAll(vectors);
        var cv = new CompressedVectors(pq, compressed);
        // Write compressed vectors
        File cvFile = File.createTempFile("pqtest", ".cv");
        try (var out = new DataOutputStream(new FileOutputStream(cvFile))) {
            cv.write(out);
        }
        // Read compressed vectors
        try (var in = new SimpleMappedReader(cvFile.getAbsolutePath())) {
            var cv2 = CompressedVectors.load(in, 0);
            assertEquals(cv, cv2);
        }
    }
}
