package io.github.jbellis.jvector.vector.types;

import java.io.DataOutput;
import java.io.IOException;

import io.github.jbellis.jvector.disk.RandomAccessReader;

public interface VectorTypeSupport {
    VectorFloat<?> createFloatType(Object data);
    VectorFloat<?> createFloatType(int length);

    VectorFloat<?> readFloatType(RandomAccessReader r, int size) throws IOException;
    void writeFloatType(DataOutput out, VectorFloat<?> vector) throws IOException;

    VectorByte<?> createByteType(Object data);
    VectorByte<?> createByteType(int length);

    VectorByte<?> readByteType(RandomAccessReader r, int size) throws IOException;
    void writeByteType(DataOutput out, VectorByte<?> vector) throws IOException;

}
