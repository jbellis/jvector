package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.util.Bits;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

class RenumberingGraphIndex<T> implements GraphIndex<T> {
    private final OnHeapGraphIndex<T> graph;
    private final View<T> view;
    private final Map<Integer, Integer> newToOldMap;
    private final Map<Integer, Integer> oldToNewMap;

    public RenumberingGraphIndex(OnHeapGraphIndex<T> graph) {
        this.graph = graph;
        this.view = graph.getView();
        this.newToOldMap = new HashMap<>();
        this.oldToNewMap = new HashMap<>();
        int nextOrdinal = 0;
        for (int i = 0; i <= graph.getMaxNodeId(); i++) {
            if (graph.getNeighbors(i) != null) {
                oldToNewMap.put(i, nextOrdinal);
                newToOldMap.put(nextOrdinal++, i);
            }
        }
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
    public void close() {
        graph.close();
    }

    private class RenumberingView implements View<T> {
        @Override
        public NodesIterator getNeighborsIterator(int node) {
            var it = graph.getNeighbors(newToOldMap.get(node)).iterator();
            return new NodesIterator(it.size()) {
                @Override
                public int nextInt() {
                    return oldToNewMap.get(it.nextInt());
                }

                @Override
                public boolean hasNext() {
                    return it.hasNext();
                }
            };
        }

        @Override
        public int size() {
            return graph.size();
        }

        @Override
        public int entryNode() {
            return oldToNewMap.get(view.entryNode());
        }

        @Override
        public T getVector(int node) {
            return view.getVector(newToOldMap.get(node));
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public void close() {
            // no-op
        }
    }
}
