package com.github.jbellis.jvector.graph;

import java.util.List;

/**
 * An List-backed implementation of the {@link RandomAccessVectorValues} interface.
 *
 * It is acceptable to provide this class to an HnswGraphBuilder, and then continue
 * to add vectors to it as you add to the graph.
 *
 * This will be as threadsafe as the provided List.
 */
public class ListRandomAccessVectorValues implements RandomAccessVectorValues<float[]> {

    private final List<float[]> vectors;
    private final int dimension;

    /**
     * Construct a new instance of {@link ListRandomAccessVectorValues}.
     *
     * @param vectors   a (potentially mutable) list of float vectors.
     * @param dimension the dimension of the vectors.
     */
    public ListRandomAccessVectorValues(List<float[]> vectors, int dimension) {
        this.vectors = vectors;
        this.dimension = dimension;
    }

    @Override
    public int size() {
        return vectors.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        return vectors.get(targetOrd);
    }

    @Override
    public ListRandomAccessVectorValues copy() {
        // the copy method is called as a workaround for Lucene's implementations not being re-entrant.
        // if you are already re-entrant, you really don't need a new copy, but HGBuilder.build
        // explicitly checks for object identity so we'll do a shallow copy to make it happy.
        return new ListRandomAccessVectorValues(vectors, dimension);
    }
}
