package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.util.Bits;

import java.io.IOException;
import java.util.stream.IntStream;

class RenumberingGraphIndex<T> implements GraphIndex<T> {
    private final OnHeapGraphIndex graph;

    public RenumberingGraphIndex(OnHeapGraphIndex graph) {
        this.graph = graph;
    }

    @Override
    public int size() {
        return graph.size();
    }

    @Override
    public NodesIterator getNodes() {
        return NodesIterator.fromPrimitiveIterator(IntStream.range(0, size()).iterator(), size());
    }

    @Override
    public View<T> getView() {
        return new RenumberingView();
    }

    @Override
    public int maxDegree() {
        return graph.maxDegree();
    }

    @Override
    public void close() throws IOException {
        graph.close();
    }

    private class RenumberingView implements View<T> {
        @Override
        public NodesIterator getNeighborsIterator(int node) {
            return null;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public int entryNode() {
            return 0;
        }

        @Override
        public T getVector(int node) {
            return null;
        }

        @Override
        public Bits liveNodes() {
            return null;
        }

        @Override
        public void close() throws Exception {

        }
    }
}
