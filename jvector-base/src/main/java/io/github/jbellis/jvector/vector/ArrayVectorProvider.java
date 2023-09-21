package io.github.jbellis.jvector.vector;


import java.io.DataOutput;
import java.io.IOException;

import io.github.jbellis.jvector.disk.Io;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class ArrayVectorProvider implements VectorTypeSupport
{

    @Override
    public VectorFloat<?> createFloatType(Object data)
    {
        return new ArrayVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatType(int length)
    {
        return new ArrayVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatType(RandomAccessReader r, int size) throws IOException
    {
        float[] vector = new float[size];
        r.readFully(vector);
        return new ArrayVectorFloat(vector);
    }

    @Override
    public void writeFloatType(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        ArrayVectorFloat v = (ArrayVectorFloat)vector;
        Io.writeFloats(out, v.get());
    }

    @Override
    public VectorByte<?> createByteType(Object data)
    {
        return new ArrayVectorByte((byte[]) data);
    }

    @Override
    public VectorByte<?> createByteType(int length)
    {
        return new ArrayVectorByte(length);
    }

    @Override
    public VectorByte<?> readByteType(RandomAccessReader r, int size) throws IOException
    {
        byte[] vector = new byte[size];
        r.readFully(vector);
        return new ArrayVectorByte(vector);
    }

    @Override
    public void writeByteType(DataOutput out, VectorByte<?> vector) throws IOException
    {
        ArrayVectorByte v = (ArrayVectorByte) vector;
        out.write(v.get());
    }
}
