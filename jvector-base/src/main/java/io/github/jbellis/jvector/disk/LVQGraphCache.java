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

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;

public abstract class LVQGraphCache implements Accountable
{
    public static final class CachedNode {
        public final VectorFloat<?> vector;
        public final int[] neighbors;

        public CachedNode(VectorFloat<?> vector, int[] neighbors) {
            this.vector = vector;
            this.neighbors = neighbors;
        }
    }

    /** return the cached node if present, or null if not */
    public abstract CachedNode getNode(int ordinal);

    public static LVQGraphCache load(GraphIndex graph, int distance)
    {
        if (distance < 0)
            return new EmptyGraphCache();
        return new HMGraphCache(graph, distance);
    }

    public abstract long ramBytesUsed();

    private static final class EmptyGraphCache extends LVQGraphCache
    {
        @Override
        public CachedNode getNode(int ordinal) {
            return null;
        }

        @Override
        public long ramBytesUsed()
        {
            return 0;
        }
    }

    private static final class HMGraphCache extends LVQGraphCache
    {
        // Map is created on construction and never modified
        private final Int2ObjectHashMap<CachedNode> cache;
        private long ramBytesUsed = 0;

        public HMGraphCache(GraphIndex graph, int distance) {
            try (var view = graph.getView()) {
                var tmpCache = new Int2ObjectHashMap<CachedNode>();
                cacheNeighborsOf(tmpCache, (GraphIndex.ViewWithVectors) view, view.entryNode(), distance);
                // Assigning to a final value ensure it is safely published
                cache = tmpCache;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private void cacheNeighborsOf(Int2ObjectHashMap<CachedNode> tmpCache, GraphIndex.ViewWithVectors view, int ordinal, int distance) {
            // cache this node
            var it = view.getNeighborsIterator(ordinal);
            int[] neighbors = new int[it.size()];
            int i = 0;
            while (it.hasNext()) {
                neighbors[i++] = it.nextInt();
            }
            var node = new CachedNode(view.getVector(ordinal), neighbors);
            tmpCache.put(ordinal, node);
            //ramBytesUsed += 4 + RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.sizeOf(node.vector) + RamUsageEstimator.sizeOf(node.neighbors);
            // TODO FIX
            // call recursively on neighbors
            if (distance > 0) {
                for (var neighbor : neighbors) {
                    if (!tmpCache.containsKey(neighbor)) {
                        cacheNeighborsOf(tmpCache, view, neighbor, distance - 1);
                    }
                }
            }
        }


        @Override
        public CachedNode getNode(int ordinal) {
            return cache.get(ordinal);
        }

        @Override
        public long ramBytesUsed()
        {
            return ramBytesUsed;
        }
    }
}
