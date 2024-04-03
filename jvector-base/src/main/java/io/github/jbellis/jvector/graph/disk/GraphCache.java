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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import org.agrona.collections.Int2ObjectHashMap;

public abstract class GraphCache<T extends GraphCache.CachedNode> implements Accountable
{
    public static class CachedNode implements Accountable {
        public final int[] neighbors;

        public CachedNode( int[] neighbors) {
            this.neighbors = neighbors;
        }

        public long ramBytesUsed() {
            return RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (long) neighbors.length * Integer.BYTES;
        }
    }

    /** return the cached node if present, or null if not */
    public abstract T getNode(int ordinal);

    public static <U extends OnDiskView<V>, V extends CachedNode> GraphCache<V> load(OnDiskGraphIndex<U, V> graph, int distance)
    {
        if (distance < 0)
            return new EmptyGraphCache<>();
        return new HMGraphCache<>(graph, distance);
    }

    public abstract long ramBytesUsed();

    private static final class EmptyGraphCache<U extends CachedNode> extends GraphCache<U>
    {
        @Override
        public U getNode(int ordinal) {
            return null;
        }

        @Override
        public long ramBytesUsed()
        {
            return 0;
        }
    }

    private static final class HMGraphCache<T extends OnDiskView<U>, U extends CachedNode> extends GraphCache<U>
    {
        // Map is created on construction and never modified
        private final Int2ObjectHashMap<U> cache;
        private long ramBytesUsed = 0;

        public HMGraphCache(OnDiskGraphIndex<T, U> graph, int distance) {
            try (var view = graph.getView()) {
                var tmpCache = new Int2ObjectHashMap<U>();
                cacheNeighborsOf(tmpCache, view, view.entryNode(), distance);
                // Assigning to a final value ensure it is safely published
                cache = tmpCache;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private void cacheNeighborsOf(Int2ObjectHashMap<U> tmpCache, OnDiskView<U> view, int ordinal, int distance) {
            // cache this node
            var it = view.getNeighborsIterator(ordinal);
            int[] neighbors = new int[it.size()];
            int i = 0;
            while (it.hasNext()) {
                neighbors[i++] = it.nextInt();
            }
            var node = view.loadCachedNode(ordinal, neighbors);
            tmpCache.put(ordinal, node);
            // ignores internal Map overhead but that should be negligible compared to the node contents
            ramBytesUsed += Integer.BYTES + RamUsageEstimator.NUM_BYTES_OBJECT_REF + node.ramBytesUsed();

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
        public U getNode(int ordinal) {
            return cache.get(ordinal);
        }

        @Override
        public long ramBytesUsed()
        {
            return ramBytesUsed;
        }
    }
}
