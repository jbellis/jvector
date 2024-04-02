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
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;

public class CachingGraphIndex implements GraphIndex, AutoCloseable, Accountable
{
    private static final int CACHE_DISTANCE = 3;

    private final GraphCache cache;
    private final OnDiskGraphIndex graph;

    public CachingGraphIndex(OnDiskGraphIndex graph)
    {
        this(graph, CACHE_DISTANCE);
    }

    public CachingGraphIndex(OnDiskGraphIndex graph, int cacheDistance)
    {
        this.graph = graph;
        this.cache = GraphCache.load(graph, cacheDistance);
    }

    @Override
    public int size() {
        return graph.size();
    }

    @Override
    public NodesIterator getNodes() {
        return graph.getNodes();
    }

    @Override
    public View getView() {
        return new CachedView(graph.getView());
    }

    @Override
    public int maxDegree() {
        return graph.maxDegree();
    }

    @Override
    public long ramBytesUsed() {
        return graph.ramBytesUsed() + cache.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
        graph.close();
    }

    @Override
    public String toString() {
        return String.format("CachingGraphIndex(graph=%s)", graph);
    }

    private class CachedView implements RerankingView {
        private final RerankingView view;

        public CachedView(RerankingView view) {
            this.view = view;
        }

        @Override
        public NodesIterator getNeighborsIterator(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return new NodesIterator.ArrayNodesIterator(cached.neighbors, cached.neighbors.length);
            }
            return view.getNeighborsIterator(node);
        }

        @Override
        public int dimension() {
            return graph.dimension;
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues copy() {
            // we would need to be able to copy the View to do this correctly, but it's simple to just
            // avoid calling copy() for non-shared RAVV instances like this
            throw new UnsupportedOperationException();
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return cached.vector;
            }
            return view.getVector(node);
        }

        @Override
        public void getVectorInto(int node, VectorFloat<?> vector, int offset) {
            var cached = cache.getNode(node);
            if (cached != null) {
                vector.copyFrom(cached.vector, 0, offset, cached.vector.length());
                return;
            }
            view.getVectorInto(node, vector, offset);
        }

        @Override
        public int size() {
            return view.size();
        }

        @Override
        public int entryNode() {
            return view.entryNode();
        }

        @Override
        public Bits liveNodes() {
            return view.liveNodes();
        }

        @Override
        public void close() throws Exception {
            view.close();
        }
    }
}
