package io.github.jbellis.jvector.vector;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentScope;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.Objects;

import io.github.jbellis.jvector.vector.types.VectorFloat;

final public class OffHeapVectorFloat implements VectorFloat<MemorySegment>
{
    private final MemorySegment segment;
    private final int length;

    OffHeapVectorFloat(int length) {
        this.segment = MemorySegment.allocateNative(MemoryLayout.sequenceLayout(length, ValueLayout.JAVA_FLOAT), SegmentScope.auto());
        this.length = length;
    }

    OffHeapVectorFloat(FloatBuffer buffer) {
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
        return 0;
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
    public Float get(int n)
    {
        return segment.get(ValueLayout.JAVA_FLOAT, offset(n));
    }

    @Override
    public void set(int n, Float value)
    {
        segment.set(ValueLayout.JAVA_FLOAT, offset(n), value);
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
        return Objects.equals(segment.asByteBuffer(), that.segment.asByteBuffer());
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(segment.asByteBuffer());
    }
}
