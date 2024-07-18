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

package io.github.jbellis.jvector;

import io.github.jbellis.jvector.vector.MemorySegmentVectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

public class GPUTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void main(String[] args) throws IOException {
        LibraryLoader.loadJvector();
        NativeGpuOps.initialize();

        MemorySegment dataset = NativeGpuOps.load_pq_vectors(MemorySegment.ofArray("/home/jonathan/Projects/jvector/cohere.pqv".getBytes()));
        int dim = 1024; // Assuming the dimension is 1024 as in the C++ code

        // Test with ones vector
        float[] ones = new float[dim];
        Arrays.fill(ones, 1.0f);
        var v = vts.createFloatVector(ones);
        testWithVector(dataset, (MemorySegmentVectorFloat) v);

        // Benchmark with random vectors
        benchmarkRandomVectors(dataset, dim);

        NativeGpuOps.free_jpq_dataset(dataset);
    }

    private static void testWithVector(MemorySegment dataset, MemorySegmentVectorFloat q) {
        MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, q.get());

        int[] nodeIds = new int[10];
        for (int i = 0; i < nodeIds.length; i++) {
            nodeIds[i] = i;
        }

        var similarities = vts.createFloatVector(nodeIds.length);
        NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                 MemorySegment.ofArray(nodeIds),
                                                 ((MemorySegmentVectorFloat) similarities).get(),
                                                 nodeIds.length);

        System.out.println("Similarities with ones");
        for (int i = 0; i < similarities.length(); i++) {
            System.out.println(similarities.get(i));
        }

        NativeGpuOps.free_adc_query(prepared);
    }

    private static void benchmarkRandomVectors(MemorySegment dataset, int dim) {
        Random random = new Random();
        int numQueries = 100_000;
        int numNodes = 200;

        long startTime = System.nanoTime();

        var queryVector = vts.createFloatVector(dim);
        int[] nodeIds = new int[numNodes];
        var similarities = vts.createFloatVector(numNodes);

        for (int i = 0; i < numQueries; i++) {
            // Generate random query vector
            for (int j = 0; j < dim; j++) {
                queryVector.set(j, random.nextFloat() * 2 - 1); // Random float between -1 and 1
            }

            // Generate random node IDs
            for (int j = 0; j < numNodes; j++) {
                nodeIds[j] = random.nextInt(100000);
            }

            MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, ((MemorySegmentVectorFloat) queryVector).get());
            NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                     MemorySegment.ofArray(nodeIds),
                                                     ((MemorySegmentVectorFloat) similarities).get(),
                                                     numNodes);
            NativeGpuOps.free_adc_query(prepared);
        }

        long endTime = System.nanoTime();
        double elapsedSeconds = (endTime - startTime) / 1e9;

        System.out.printf("Time elapsed: %.3f seconds%n", elapsedSeconds);
    }
}
