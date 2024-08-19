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

import io.github.jbellis.jvector.graph.AcceleratedIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GPUCagraIndex implements AcceleratedIndex.ExternalIndex {
    // DEMOFIXME workaround for CAGRA's broken concurrency
    // (single thread executor is modestly better than using a synchronized block in my testing)
    private final ExecutorService nativeExecutor = Executors.newSingleThreadExecutor();

    private final MemorySegment index;
    private final ThreadLocal<MemorySegmentVectorFloat> reusableQuery;
    private final ThreadLocal<MemorySegmentByteSequence> reusableIds;
    private final int TOP_K_MAX = 1000;

    public GPUCagraIndex(MemorySegment index, int dimension) {
        this.index = index;
        this.reusableQuery = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(dimension).reinterpret(dimension * 4)));
        this.reusableIds = ThreadLocal.withInitial(() -> new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(TOP_K_MAX).reinterpret(TOP_K_MAX * 4)));
    }

    public static GPUCagraIndex build(RandomAccessVectorValues ravv) {
        MemorySegment builder = NativeGpuOps.create_cagra_builder(ravv.size(), ravv.dimension());
        for (int i = 0; i < ravv.size(); i++) {
            NativeGpuOps.add_node(builder, ((MemorySegmentVectorFloat) ravv.getVector(i)).get());
        }
        var index = NativeGpuOps.build_cagra_index(builder);
        return new GPUCagraIndex(index, ravv.dimension());
    }

    @Override
    public NodesIterator search(VectorFloat<?> query, int topK) {
        if (topK > TOP_K_MAX) {
            throw new IllegalArgumentException("rerankK must be <= " + TOP_K_MAX);
        }
        // DEMOFIXME: use actual size? Can Cagra return too few results?
        MemorySegmentVectorFloat unifiedQuery = reusableQuery.get();
        unifiedQuery.copyFrom(query, 0, 0, query.length());
        MemorySegment ms = null;
        try {
            ms = nativeExecutor.submit(() -> NativeGpuOps.search_cagra_index(index, unifiedQuery.get(), topK)).get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
        ms = ms.reinterpret(topK * Integer.BYTES);
        MemorySegmentByteSequence ids = new MemorySegmentByteSequence(ms);
        int size = 0;
        // DEMOFIXME: Is there a better way to tell if we didn't get the full number of results?
        // We've seen ids of both -1 and MAX_VALUE - 1 that seem to indicate results are done.
        // Investigate and figure out this comment.
        for (int i = 0; i < ids.length() / 4; i++) {
            var id = ids.getLittleEndianInt(i);
            if (id != -1 && id != Integer.MAX_VALUE) {
                size++;
            } else {
                // this is how CAGRA indicates under the requested number of nodes
                break;
            }
        }
        // iterate over LE ints in ids
        return new NodesIterator(size) {
            int i = 0;
            @Override
            public int nextInt() {
                return ids.getLittleEndianInt(i++);
            }

            @Override
            public boolean hasNext() {
                return i < size;
            }
        };
    }

    @Override
    public int size() {
        // DEMOFIXME: should probably call in to native size, this is hardcoded to cohere dataset
        return 99740;
    }

    @Override
    public void save(String filename) {
        NativeGpuOps.save_cagra_index(index, MemorySegment.ofArray(filename.getBytes()));
    }

    public static AcceleratedIndex.ExternalIndex load(String filename) {
        // DEMOFIXME can we derive dimension from the index?
        return new GPUCagraIndex(NativeGpuOps.load_cagra_index(MemorySegment.ofArray(filename.getBytes())), 1024);
    }
}
