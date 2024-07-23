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

import io.github.jbellis.jvector.vector.MemorySegmentByteSequence;
import io.github.jbellis.jvector.vector.MemorySegmentVectorFloat;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

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
        // benchmarkRaw(dataset);
        benchmarkADC(dataset);

        NativeGpuOps.free_jpq_dataset(dataset);

        // Run CAGRA benchmark
        NativeGpuOps.call_cagra_demo();
    }

    private static void testWithOnes(MemorySegment dataset) {
        float[] ones = new float[DIM];
        Arrays.fill(ones, 1.0f);
        MemorySegmentVectorFloat q = (MemorySegmentVectorFloat) vts.createFloatVector(ones);

        int nodeIdCount = 10;
        var nodeIds = new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(nodeIdCount).reinterpret(nodeIdCount * Integer.BYTES));
        for (int i = 0; i < nodeIdCount; i++) {
            nodeIds.setLittleEndianInt(i, i);
        }

        var similarities = new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(nodeIdCount).reinterpret(nodeIdCount * Float.BYTES));
        // Compute similarities without ADC
        MemorySegment query = NativeGpuOps.prepare_query(dataset, q.get());
        NativeGpuOps.compute_dp_similarities(query,
                                             nodeIds.get(),
                                             similarities.get(),
                                             nodeIdCount);
        NativeGpuOps.free_query(query);
        System.out.println("Similarity with ones (raw):");
        for (int i = 0; i < similarities.length(); i++) {
            System.out.println(similarities.get(i));
        }

        // Compute similarities with ADC
        // (put the ones in the second query slot to test that it's working)
        var multiQ = (MemorySegmentVectorFloat) vts.createFloatVector(DIM * 2);
        multiQ.copyFrom(q, 0, DIM, DIM);
        MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, multiQ.get(), 2);
        var multiIds = new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(2 * nodeIdCount).reinterpret(2 * nodeIdCount * Integer.BYTES));
        for (int i = 0; i < nodeIdCount; i++) {
            multiIds.setLittleEndianInt(i + nodeIdCount, i);
        }
        var multiSimilarities = new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(2 * nodeIdCount).reinterpret(2 * nodeIdCount * Float.BYTES));
        NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                 multiIds.get(),
                                                 multiSimilarities.get(),
                                                 nodeIdCount);
        NativeGpuOps.free_adc_query(prepared);
        System.out.println("Similarity with ones (ADC):");
        for (int i = nodeIdCount; i < multiSimilarities.length(); i++) {
            System.out.println(multiSimilarities.get(i));
        }
    }

    private static void benchmarkRaw(MemorySegment dataset) {
        var R = ThreadLocalRandom.current();
        int nodeIdCount = 200;
        MemorySegmentVectorFloat similarities = new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(nodeIdCount).reinterpret(nodeIdCount * Float.BYTES));
        MemorySegmentVectorFloat q = (MemorySegmentVectorFloat) vts.createFloatVector(DIM);
        MemorySegmentByteSequence nodeIds = new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(nodeIdCount).reinterpret(nodeIdCount * Integer.BYTES));

        long startTime = System.nanoTime();
        for (int i = 0; i < 1000_000; i++) {
            // Generate random query vector
            for (int j = 0; j < DIM; j++) {
                q.set(j, R.nextFloat() * 2 - 1);
            }

            // Generate random node IDs
            for (int j = 0; j < nodeIdCount; j++) {
                nodeIds.setLittleEndianInt(j, R.nextInt(100_000));
            }

            MemorySegment query = NativeGpuOps.prepare_query(dataset, q.get());
            NativeGpuOps.compute_dp_similarities(query,
                                                 nodeIds.get(),
                                                 similarities.get(),
                                                 nodeIdCount);
            NativeGpuOps.free_query(query);
        }
        long endTime = System.nanoTime();
        System.out.printf("Time elapsed for 10000x 200 (Raw): %.3f seconds%n", (endTime - startTime) / 1e9);
    }

    private static void benchmarkADC(MemorySegment dataset) {
        var R = ThreadLocalRandom.current();
        int nodesPerQuery = 32;
        int queriesPerBatch = 50;
        var multiSimilarities = new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(nodesPerQuery * queriesPerBatch)
                                                                     .reinterpret(nodesPerQuery * queriesPerBatch * Float.BYTES));
        var multiQ = (MemorySegmentVectorFloat) vts.createFloatVector(queriesPerBatch * DIM);
        var multiIds = new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(nodesPerQuery * queriesPerBatch)
                                                             .reinterpret(nodesPerQuery * queriesPerBatch * Integer.BYTES));

        long startTime = System.nanoTime();
        for (int i = 0; i < 100_000; i++) {
            MemorySegment prepared = NativeGpuOps.prepare_adc_query(dataset, multiQ.get(), nodesPerQuery);

            for (int j = 0; j < queriesPerBatch; j++) {
                // Generate random query vectors
                for (int k = 0; k < DIM; k++) {
                    multiQ.set(j * DIM + k, R.nextFloat() * 2 - 1);
                }
                // Generate random node IDs
                for (int k = 0; k < nodesPerQuery; k++) {
                    multiIds.setLittleEndianInt(j * nodesPerQuery + k, R.nextInt(100_000));
                }

                NativeGpuOps.compute_dp_similarities_adc(prepared,
                                                         multiIds.get(),
                                                         multiSimilarities.get(),
                                                         nodesPerQuery);
            }

            NativeGpuOps.free_adc_query(prepared);
        }
        long endTime = System.nanoTime();
        System.out.printf("Time elapsed for 1000x 50x32 (ADC): %.3f seconds%n", (endTime - startTime) / 1e9);
    }
}
