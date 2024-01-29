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
import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public abstract class GraphCache implements Accountable
{
    public static final class CachedNode {
        public final float[] vector;
        public final int[] neighbors;

        public CachedNode(float[] vector, int[] neighbors) {
            this.vector = vector;
            this.neighbors = neighbors;
        }
    }

    /** return the cached node if present, or null if not */
    public abstract CachedNode getNode(int ordinal);

    public static GraphCache load(GraphIndex<float[]> graph, int distance) throws IOException
    {
        if (distance < 0)
            return new EmptyGraphCache();
        return new HMGraphCache(graph, distance);
    }

    public abstract long ramBytesUsed();

    private static final class EmptyGraphCache extends GraphCache
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

    private static final class HMGraphCache extends GraphCache
    {
        // Map is created on construction and never modified
        private final Map<Integer, CachedNode> cache;
        private long ramBytesUsed = 0;

        public HMGraphCache(GraphIndex<float[]> graph, int distance) {
            try (var view = graph.getView()) {
                HashMap<Integer, CachedNode> tmpCache = new HashMap<>();
                cacheNeighborsOf(tmpCache, view, view.entryNode(), distance);
                // Assigning to a final value ensure it is safely published
                cache = tmpCache;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        private void cacheNeighborsOf(HashMap<Integer, CachedNode> tmpCache, GraphIndex.View<float[]> view, int ordinal, int distance) {
            // cache this node
            var it = view.getNeighborsIterator(ordinal);
            int[] neighbors = new int[it.size()];
            int i = 0;
            while (it.hasNext()) {
                neighbors[i++] = it.next();
            }
            var node = new CachedNode(view.getVector(ordinal), neighbors);
            tmpCache.put(ordinal, node);
            ramBytesUsed += RamUsageEstimator.HASHTABLE_RAM_BYTES_PER_ENTRY + RamUsageEstimator.sizeOf(node.vector) + RamUsageEstimator.sizeOf(node.neighbors);

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
