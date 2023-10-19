package io.github.jbellis.jvector.example.util;

import java.util.ArrayList;
import java.util.List;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

public class UpdatableRandomAccessVectorValues implements RandomAccessVectorValues<float[]> {
    private final List<float[]> data;
    private final int dimensions;

    public UpdatableRandomAccessVectorValues(int dimensions) {
        this.data = new ArrayList<>(1024);
        this.dimensions = dimensions;
    }

    public void add(float[] vector) {
        data.add(vector);
    }

    @Override
    public int size() {
        return data.size();
    }

    @Override
    public int dimension() {
        return dimensions;
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        return data.get(targetOrd);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues<float[]> copy() {
        return this;
    }
}
