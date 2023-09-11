package com.github.jbellis.jvector.graph;

import com.github.jbellis.jvector.annotations.Unshared;

import java.util.List;

/**
 * A List-backed implementation of the {@link RandomAccessVectorValues} interface.
 *
 * It is acceptable to provide this class to a GraphBuilder, and then continue
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
    @Unshared
    public float[] vectorValue(int targetOrd) {
        return vectors.get(targetOrd);
    }

    @Override
    public ListRandomAccessVectorValues copy() {
        // our vectorValue is Unshared, but copy anyway in case the underlying List is not threadsafe
        return new ListRandomAccessVectorValues(List.copyOf(vectors), dimension);
    }
}
