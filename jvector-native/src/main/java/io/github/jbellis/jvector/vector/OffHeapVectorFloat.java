package io.github.jbellis.jvector.vector;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

import io.github.jbellis.jvector.vector.types.VectorFloat;

final public class OffHeapVectorFloat implements VectorFloat<MemorySegment>
{
    private final MemorySegment segment;
    private final ByteBuffer buffer;
    private final int length;

    OffHeapVectorFloat(int length) {
        this.buffer = ByteBuffer.allocateDirect(length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        this.segment = MemorySegment.ofBuffer(buffer);
        this.length = length;
    }

    OffHeapVectorFloat(ByteBuffer buffer) {
        this.buffer = buffer;
        this.segment = MemorySegment.ofBuffer(buffer);
        this.length = buffer.remaining();
    }

    OffHeapVectorFloat(float[] data) {
        this(data.length);
        segment.copyFrom(MemorySegment.ofArray(data));
    }

    @Override
    public long ramBytesUsed()
    {
        return MemoryLayout.sequenceLayout(length, ValueLayout.JAVA_FLOAT).byteSize();
    }

    @Override
    public VectorEncoding type()
    {
        return VectorEncoding.FLOAT32;
    }

    @Override
    public MemorySegment get()
    {
        return segment;
    }

    @Override
    public float get(int n)
    {
        return segment.getAtIndex(ValueLayout.JAVA_FLOAT, n);
    }

    @Override
    public void set(int n, float value)
    {
        segment.setAtIndex(ValueLayout.JAVA_FLOAT, n, value);
    }

    @Override
    public int length() {
        return length;
    }

    @Override
    public int offset(int i)
    {
        return i * Float.BYTES;
    }

    @Override
    public VectorFloat<MemorySegment> copy()
    {
        OffHeapVectorFloat copy = new OffHeapVectorFloat(length());
        copy.copyFrom(this, 0, 0, length());
        return copy;
    }

    @Override
    public void copyFrom(VectorFloat<?> src, int srcOffset, int destOffset, int length)
    {
        OffHeapVectorFloat csrc = (OffHeapVectorFloat) src;
        segment.asSlice((long) destOffset * Float.BYTES, (long) length * Float.BYTES)
                .copyFrom(csrc.segment.asSlice((long) srcOffset * Float.BYTES, (long) length * Float.BYTES));
    }

    @Override
    public float[] array()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OffHeapVectorFloat that = (OffHeapVectorFloat) o;
        return Objects.equals(buffer, that.buffer);
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(buffer);
    }
}
