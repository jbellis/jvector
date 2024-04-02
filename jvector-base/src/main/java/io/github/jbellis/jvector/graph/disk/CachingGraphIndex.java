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

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;

import java.io.IOException;

public class CachingGraphIndex implements GraphIndex, Accountable
{
    private static final int CACHE_DISTANCE = 3;

    private final GraphCache cache_;
    private final OnDiskGraphIndex graph;

    public CachingGraphIndex(OnDiskGraphIndex graph)
    {
        this(graph, CACHE_DISTANCE);
    }

    public CachingGraphIndex(OnDiskGraphIndex graph, int cacheDistance)
    {
        this.graph = graph;
        this.cache_ = GraphCache.load(graph, cacheDistance);
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
    public RerankingView getView() {
        return graph.getView().cachedWith(cache_);
    }

    @Override
    public int maxDegree() {
        return graph.maxDegree();
    }

    @Override
    public long ramBytesUsed() {
        return graph.ramBytesUsed() + cache_.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
        graph.close();
    }

    @Override
    public String toString() {
        return String.format("CachingGraphIndex(graph=%s)", graph);
    }

    public static abstract class View implements RerankingView {
        private final GraphCache cache;
        protected final RerankingView view;

        public View(GraphCache cache, RerankingView view) {
            this.cache = cache;
            this.view = view;
        }

        @Override
        public NodesIterator getNeighborsIterator(int ordinal) {
            var node = getCachedNode(ordinal);
            if (node != null) {
                return new NodesIterator.ArrayNodesIterator(node.neighbors, node.neighbors.length);
            }
            return view.getNeighborsIterator(ordinal);
        }

        protected GraphCache.CachedNode getCachedNode(int ordinal) {
            return cache.getNode(ordinal);
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
