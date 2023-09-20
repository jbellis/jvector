package io.github.jbellis.jvector.vector.types;

public interface VectorByte<T> extends VectorType<Byte, T>
{
    //Hack till non-array support is added
    byte[] array();

    void copyFrom(VectorByte<?> src, int srcOffset, int destOffset, int length);

}
