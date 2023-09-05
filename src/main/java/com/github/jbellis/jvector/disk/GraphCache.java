package com.github.jbellis.jvector.disk;

import com.github.jbellis.jvector.graph.GraphIndex;
import com.github.jbellis.jvector.util.Accountable;
import com.github.jbellis.jvector.util.RamUsageEstimator;
import org.cliffc.high_scale_lib.NonBlockingHashMapLong;

import java.io.IOException;

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
        if (distance <= 0)
            return new EmptyGraphCache();
        return new NBHMGraphCache(graph, distance);
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

    private static final class NBHMGraphCache extends GraphCache
    {
        private final NonBlockingHashMapLong<CachedNode> cache = new NonBlockingHashMapLong<>();
        private long ramBytesUsed = 0;

        public NBHMGraphCache(GraphIndex<float[]> graph, int distance) {
            var view = graph.getView();
            cacheNeighborsOf(view, view.entryNode(), distance);
        }

        private void cacheNeighborsOf(GraphIndex.View<float[]> view, int ordinal, int distance) {
            // cache this node
            var it = view.getNeighborsIterator(ordinal);
            int[] neighbors = new int[it.size()];
            int i = 0;
            while (it.hasNext()) {
                neighbors[i++] = it.next();
            }
            var node = new CachedNode(view.getVector(ordinal), neighbors);
            cache.put(ordinal, node);
            ramBytesUsed += RamUsageEstimator.HASHTABLE_RAM_BYTES_PER_ENTRY + RamUsageEstimator.sizeOf(node.vector) + RamUsageEstimator.sizeOf(node.neighbors);

            // call recursively on neighbors
            if (distance > 0) {
                for (var neighbor : neighbors) {
                    if (!cache.containsKey(neighbor)) {
                        cacheNeighborsOf(view, neighbor, distance - 1);
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
