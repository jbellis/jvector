package io.github.jbellis.jvector.vector;

import java.util.Arrays;

import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.types.VectorByte;

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
    public byte get(int n) {
        return data[n];
    }

    @Override
    public void set(int n, byte value) {
        data[n] = value;
    }

    @Override
    public int length() {
        return data.length;
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

    @Override
    public void copyFrom(VectorByte<?> src, int srcOffset, int destOffset, int length) {
        ArrayVectorByte csrc = (ArrayVectorByte) src;
        System.arraycopy(csrc.data, srcOffset, data, destOffset, length);
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ArrayVectorByte that = (ArrayVectorByte) o;
        return Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode()
    {
        return Arrays.hashCode(data);
    }
}
