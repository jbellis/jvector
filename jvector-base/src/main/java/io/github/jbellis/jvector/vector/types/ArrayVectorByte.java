package io.github.jbellis.jvector.vector.types;

import java.util.Arrays;

import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorEncoding;

final public class ArrayVectorByte implements VectorByte<byte[]>
{
    private final byte[] data;

    ArrayVectorByte(int length) {
        this.data = new byte[length];
    }

    ArrayVectorByte(byte[] data) {
        this.data = data;
    }

    @Override
    public VectorEncoding type() {
        return VectorEncoding.BYTE;
    }

    @Override
    public byte[] get() {
        return data;
    }

    @Override
    public Byte get(int n) {
        return data[n];
    }

    @Override
    public void set(int n, Byte value) {
        data[n] = value;
    }

    @Override
    public int length() {
        return data.length;
    }

    @Override
    public int offset() {
        return 0;
    }

    @Override
    public ArrayVectorByte copy() {
        return new ArrayVectorByte(Arrays.copyOf(data, data.length));
    }

    @Override
    public long ramBytesUsed() {
        return RamUsageEstimator.sizeOf(data) + RamUsageEstimator.shallowSizeOfInstance(VectorByte.class);
    }

    @Override
    public byte[] array()
    {
        return data;
    }
}
