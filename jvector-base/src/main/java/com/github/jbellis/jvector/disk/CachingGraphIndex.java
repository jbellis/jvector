package com.github.jbellis.jvector.disk;

import com.github.jbellis.jvector.graph.GraphIndex;
import com.github.jbellis.jvector.graph.NodesIterator;
import com.github.jbellis.jvector.util.Accountable;

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
    public void close() {
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
    }
}
