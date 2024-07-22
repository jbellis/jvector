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

public class GPUCagraIndex implements AcceleratedIndex.ExternalIndex {
    private final MemorySegment index;
    private final int size;
    private final ThreadLocal<MemorySegmentVectorFloat> reusableQuery;
    private final ThreadLocal<MemorySegmentByteSequence> reusableIds;

    public GPUCagraIndex(RandomAccessVectorValues ravv) {
        MemorySegment builder = NativeGpuOps.create_cagra_builder(ravv.size(), ravv.dimension());
        for (int i = 0; i < ravv.size(); i++) {
            NativeGpuOps.add_node(builder, ((MemorySegmentVectorFloat) ravv.getVector(i)).get());
        }
        index = NativeGpuOps.build_cagra_index(builder);
        size = ravv.size();
        this.reusableQuery = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(ravv.dimension()).reinterpret(ravv.dimension() * 4)));
        this.reusableIds = ThreadLocal.withInitial(() -> new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(ravv.size()).reinterpret(ravv.size() * 4)));
    }

    @Override
    public NodesIterator search(VectorFloat<?> query, int rerankK) {
        // DEMOFIXME: use actual size? Can Cagra return too few results?
        MemorySegmentVectorFloat unifiedQuery = reusableQuery.get();
        unifiedQuery.copyFrom(query, 0, 0, query.length());
        MemorySegment ms = NativeGpuOps.search_cagra_index(index, unifiedQuery.get(), rerankK);
        ms = ms.reinterpret(rerankK * Integer.BYTES);
        MemorySegmentByteSequence ids = new MemorySegmentByteSequence(ms);
        int size = 0;
        // DEMOFIXME: Is there a better way to tell if we didn't get the full number of results?
        // We've seen ids of both -1 and MAX_VALUE - 1 that seem to indicate results are done.
        // Investigate and figure out this comment.
        for (int i = 0; i < ids.length() / 4; i++) {
            var id = ids.getLittleEndianInt(i);
            if (id != -1 && id != Integer.MAX_VALUE) {
                size++;
                break;
            } else if (id == -1) {
                System.out.println("returned -1 at index " + i);
            } else if (id == Integer.MAX_VALUE) {
                System.out.println("returned Integer.MAX_VALUE at index " + i);
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
        // DEMOFIXME: should probably call in to native size
        return size;
    }
}
