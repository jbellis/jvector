package io.github.jbellis.jvector.vector.types;

public interface VectorFloat<T> extends VectorType<Float, T>
{
    @Override
    VectorFloat<T> copy();

    void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length);

    float get(int i);

    void set(int i, float value);

    //Hack till non-array support is added
    float[] array();

}
