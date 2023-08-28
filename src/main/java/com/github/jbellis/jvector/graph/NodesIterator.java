package com.github.jbellis.jvector.graph;

import java.util.PrimitiveIterator;

/**
 * Iterator over the graph nodes on a certain level, Iterator also provides the size – the total
 * number of nodes to be iterated over. The nodes are NOT guaranteed to be presented in any
 * particular order.
 */
public abstract class NodesIterator implements PrimitiveIterator.OfInt {
    public static final int NO_MORE_NEIGHBORS = Integer.MAX_VALUE;
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
}
