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
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

public class GPUTest {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int DIM = 1024;

    public static void main(String[] args) throws IOException {
        // unnecessary since VTS calls it
        // LibraryLoader.loadJvector();

        MemorySegment dataset = NativeGpuOps.load_pq_vectors(MemorySegment.ofArray("cohere.pqv".getBytes()));

        // Test with ones vector
        testWithOnes(dataset);

        // Benchmark random queries
        benchmarkRaw(dataset);
        benchmarkADC(dataset);

        NativeGpuOps.free_jpq_dataset(dataset);
    }

    private static void testWithOnes(MemorySegment dataset) {
        float[] ones = new float[DIM];
        Arrays.fill(ones, 1.0f);
        MemorySegmentVectorFloat q = (MemorySegmentVectorFloat) vts.createFloatVector(ones);

        int[] nodeIds = new int[10];
        for (int i = 0; i < nodeIds.length; i++) {
            nodeIds[i] = i;
        }

        var similarities = vts.createFloatVector(nodeIds.length);

        // Compute similarities without ADC
        MemorySegment query = NativeGpuOps.prepare_query(dataset, q.get());
        NativeGpuOps.compute_dp_similarities(query,
                                             MemorySegment.ofArray(nodeIds),
                                             ((MemorySegmentVectorFloat) similarities).get(),
                                             nodeIds.length);
        NativeGpuOps.free_query(query);
        System.out.println("Similarity with ones (raw):");
        for (int i = 0; i < similarities.length(); i++) {
            System.out.println(similarities.get(i));
        }

        // Compute similarities with ADC
        MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, q.get());
        similarities = vts.createFloatVector(nodeIds.length);
        NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                 MemorySegment.ofArray(nodeIds),
                                                 ((MemorySegmentVectorFloat) similarities).get(),
                                                 nodeIds.length);
        NativeGpuOps.free_adc_query(prepared);
        System.out.println("Similarity with ones (ADC):");
        for (int i = 0; i < similarities.length(); i++) {
            System.out.println(similarities.get(i));
        }
    }

    private static void benchmarkRaw(MemorySegment dataset) {
        var R = ThreadLocalRandom.current();
        int[] nodeIds = new int[200];
        MemorySegmentVectorFloat similarities = (MemorySegmentVectorFloat) vts.createFloatVector(nodeIds.length);
        MemorySegmentVectorFloat q = (MemorySegmentVectorFloat) vts.createFloatVector(DIM);

        long startTime = System.nanoTime();
        for (int i = 0; i < 1000_000; i++) {
            // Generate random query vector
            for (int j = 0; j < DIM; j++) {
                q.set(j, R.nextFloat() * 2 - 1);
            }

            // Generate random node IDs
            for (int j = 0; j < nodeIds.length; j++) {
                nodeIds[j] = R.nextInt(100_000);
            }

            MemorySegment query = NativeGpuOps.prepare_query(dataset, q.get());
            NativeGpuOps.compute_dp_similarities(query,
                                                 MemorySegment.ofArray(nodeIds),
                                                 similarities.get(),
                                                 nodeIds.length);
            NativeGpuOps.free_query(query);
        }
        long endTime = System.nanoTime();
        System.out.printf("Time elapsed for 10000x 200 (Raw): %.3f seconds%n", (endTime - startTime) / 1e9);
    }

    private static void benchmarkADC(MemorySegment dataset) {
        var R = ThreadLocalRandom.current();
        int[] nodeIds = new int[32];  // Changed to 32 as per the ADC benchmark in C++
        MemorySegmentVectorFloat similarities = (MemorySegmentVectorFloat) vts.createFloatVector(nodeIds.length);
        MemorySegmentVectorFloat q = (MemorySegmentVectorFloat) vts.createFloatVector(DIM);

        long startTime = System.nanoTime();
        for (int i = 0; i < 100_000; i++) {
            // Generate random query vector
            for (int j = 0; j < DIM; j++) {
                q.set(j, R.nextFloat() * 2 - 1);
            }

            MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, q.get());

            for (int j = 0; j < 50; j++) {
                // Generate random node IDs
                for (int k = 0; k < nodeIds.length; k++) {
                    nodeIds[k] = R.nextInt(100_000);
                }

                NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                         MemorySegment.ofArray(nodeIds),
                                                         similarities.get(),
                                                         nodeIds.length);
            }

            NativeGpuOps.free_adc_query(prepared);
        }
        long endTime = System.nanoTime();
        System.out.printf("Time elapsed for 1000x 50x32 (ADC): %.3f seconds%n", (endTime - startTime) / 1e9);
    }
}
