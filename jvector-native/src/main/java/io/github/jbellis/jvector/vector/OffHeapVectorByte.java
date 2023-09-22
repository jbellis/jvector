package io.github.jbellis.jvector.vector;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorType;

public class OffHeapVectorByte implements VectorByte<MemorySegment>
{
    private final ByteBuffer buffer;
    private final MemorySegment segment;

    OffHeapVectorByte(int length) {
        this.buffer = ByteBuffer.allocateDirect(length).order(ByteOrder.LITTLE_ENDIAN);
        this.segment = MemorySegment.ofBuffer(buffer);
    }

    OffHeapVectorByte(ByteBuffer data) {
        this.buffer = data;
        this.segment = MemorySegment.ofBuffer(data);
    }

    OffHeapVectorByte(byte[] data) {
        this(data.length);
        segment.copyFrom(MemorySegment.ofArray(data));
    }

    @Override
    public long ramBytesUsed()
    {
         return MemoryLayout.sequenceLayout(buffer.remaining(), ValueLayout.JAVA_BYTE).byteSize();
    }

    @Override
    public byte[] array()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyFrom(VectorByte<?> src, int srcOffset, int destOffset, int length)
    {
        OffHeapVectorByte csrc = (OffHeapVectorByte) src;
        segment.asSlice(destOffset, length).copyFrom(csrc.segment.asSlice(srcOffset));
    }

    @Override
    public VectorEncoding type() {
        return VectorEncoding.BYTE;
    }

    @Override
    public MemorySegment get() {
        return segment;
    }

    @Override
    public byte get(int n) {
        return segment.get(ValueLayout.JAVA_BYTE, n);
    }

    @Override
    public void set(int n, byte value) {
        segment.set(ValueLayout.JAVA_BYTE, n, value);
    }

    @Override
    public int length() {
        return (int) segment.byteSize();
    }

    @Override
    public VectorType<Byte, MemorySegment> copy() {
        OffHeapVectorByte copy = new OffHeapVectorByte(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OffHeapVectorByte that = (OffHeapVectorByte) o;
        return Objects.equals(buffer, that.buffer);
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(buffer);
    }
}
