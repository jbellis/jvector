package com.github.jbellis.jvector.graph;

import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;

/**
 * Iterator over the graph nodes on a certain level, Iterator also provides the size – the total
 * number of nodes to be iterated over. The nodes are NOT guaranteed to be presented in any
 * particular order.
 */
public abstract class NodesIterator implements PrimitiveIterator.OfInt {
    protected final int size;

    /**
     * Constructor for iterator based on the size
     */
    public NodesIterator(int size) {
        this.size = size;
    }

    /**
     * The number of elements in this iterator *
     */
    public int size() {
        return size;
    }

    /** Nodes iterator based on set representation of nodes. */
    public static class CollectionNodesIterator extends NodesIterator {
        Iterator<Integer> nodes;

        /** Constructor for iterator based on collection representing nodes */
        public CollectionNodesIterator(Collection<Integer> nodes) {
            super(nodes.size());
            this.nodes = nodes.iterator();
        }

        @Override
        public int nextInt() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return nodes.next();
        }

        @Override
        public boolean hasNext() {
            return nodes.hasNext();
        }
    }
}
