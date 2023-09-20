package io.github.jbellis.jvector.vector.types;

public interface VectorFloat<T> extends VectorType<Float, T>
{

    @Override
    VectorFloat<T> copy();

    void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length);

    float[] array();

}
