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
import io.github.jbellis.jvector.util.Accountable;

import java.io.IOException;
import java.io.UncheckedIOException;

public class CachingGraphIndex implements GraphIndex<float[]>, AutoCloseable, Accountable
{
    private static final int BFS_DISTANCE = 3;

    private final GraphCache cache;
    private final OnDiskGraphIndex<float[]> graph;

    public CachingGraphIndex(OnDiskGraphIndex<float[]> graph)
    {
        this.graph = graph;
        try {
            this.cache = GraphCache.load(graph, BFS_DISTANCE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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
    public View<float[]> getView() {
        return new CachedView(graph.getView());
    }

    @Override
    public int maxEdgesPerNode() {
        return graph.maxEdgesPerNode();
    }

    @Override
    public long ramBytesUsed() {
        return graph.ramBytesUsed() + cache.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
        graph.close();
    }

    private class CachedView implements View<float[]> {
        private final View<float[]> view;

        public CachedView(View<float[]> view) {
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
        public float[] getVector(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return cached.vector;
            }
            return view.getVector(node);
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
        public int[] getSortedNodes() {
            return View.super.getSortedNodes();
        }

        @Override
        public int getNeighborCount(int node) {
            return View.super.getNeighborCount(node);
        }

        @Override
        public void close() throws Exception {
            view.close();
        }
    }
}
