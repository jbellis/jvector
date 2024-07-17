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

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.vector.cnative.LibraryLoader;
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import org.junit.Test;

public class NativeGpuOpsTest {
    /*@Test
    public void testSimple() {
        // print current working directory
        LibraryLoader.loadJvector();
        NativeGpuOps.run_jpq_test_simple();
    }*/

    @Test
    public void testCohere() {
        LibraryLoader.loadJvector();
        NativeGpuOps.run_jpq_test_cohere();
    }

    /*@Test
    public void testGpuLifecycle() {
        VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        LibraryLoader.loadJvector();
        MemorySegment path = MemorySegment.ofArray("cohere.pqv".getBytes());
        MemorySegment pqvectors = NativeGpuOps.load_pq_vectors(path);
        MemorySegmentVectorFloat query = (MemorySegmentVectorFloat) vts.createFloatVector(1024);
        MemorySegment loadedQuery = NativeGpuOps.load_query(pqvectors, query.get());
        MemorySegmentVectorFloat results = (MemorySegmentVectorFloat) vts.createFloatVector(10);
        // int32 array of 10 ids 1..10
        int[] idArray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        MemorySegment ids = MemorySegment.ofArray(idArray);
        NativeGpuOps.compute_l2_similarities(loadedQuery, ids, results.get(), idArray.length);
        // print out contents of results
        for (int i = 0; i < 10; i++) {
            System.out.println("Result " + i + ": " + results.get(i));
        }
    }*/
}
