package io.github.jbellis.jvector.vector;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

public class OffHeapVectorProvider implements VectorTypeSupport
{
    @Override
    public VectorFloat<?> createFloatType(Object data)
    {
        if (data instanceof FloatBuffer)
            return new OffHeapVectorFloat((FloatBuffer) data);

        return new OffHeapVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatType(int length)
    {
        return new OffHeapVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatType(RandomAccessReader r, int size) throws IOException
    {
        float[] d = new float[size];
        r.readFully(d);
        return new OffHeapVectorFloat(d);
    }

    @Override
    public void writeFloatType(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeFloat(vector.get(i));
    }

    @Override
    public VectorByte<?> createByteType(Object data)
    {
        if (data instanceof ByteBuffer)
            return new OffHeapVectorByte((ByteBuffer) data);

        return new OffHeapVectorByte((byte[]) data);
    }

    @Override
    public VectorByte<?> createByteType(int length)
    {
        return new OffHeapVectorByte(length);
    }

    @Override
    public VectorByte<?> readByteType(RandomAccessReader r, int size) throws IOException
    {
        byte[] d = new byte[size];
        r.readFully(d);
        return new OffHeapVectorByte(d);
    }

    @Override
    public void writeByteType(DataOutput out, VectorByte<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeByte(vector.get(i));
    }
}
