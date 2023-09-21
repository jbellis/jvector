package io.github.jbellis.jvector.vector;

import java.util.Arrays;

import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.VectorFloat;

final public class ArrayVectorFloat implements VectorFloat<float[]>
{
    private final float[] data;

    ArrayVectorFloat(int length)
    {
        this.data = new float[length];
    }

    ArrayVectorFloat(float[] data)
    {
        this.data = data;
    }

    @Override
    public VectorEncoding type()
    {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public float[] get()
    {
        return data;
    }

    @Override
    public float get(int n) {
        return data[n];
    }

    @Override
    public void set(int n, float value) {
        data[n] = value;
    }

    @Override
    public int length()
    {
        return data.length;
    }

    @Override
    public VectorFloat<float[]> copy()
    {
        return new ArrayVectorFloat(Arrays.copyOf(data, data.length));
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        ArrayVectorFloat csrc = (ArrayVectorFloat) src;
        System.arraycopy(csrc.data, srcOffset, data, destOffset, length);
    }

    @Override
    public float[] array() {
        return data;
    }

    @Override
    public long ramBytesUsed()
    {
        return RamUsageEstimator.sizeOf(data) + RamUsageEstimator.shallowSizeOfInstance(ArrayVectorFloat.class);
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ArrayVectorFloat that = (ArrayVectorFloat) o;
        return Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode()
    {
        return Arrays.hashCode(data);
    }
}

