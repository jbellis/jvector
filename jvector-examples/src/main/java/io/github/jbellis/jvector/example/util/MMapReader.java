package io.github.jbellis.jvector.example.util;

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MMapReader implements RandomAccessReader {
    private final MMapBuffer buffer;
    private long position;
    private byte[] floatsScratch = new byte[0];
    private byte[] intsScratch = new byte[0];

    MMapReader(MMapBuffer buffer) {
        this.buffer = buffer;
    }

    @Override
    public void seek(long offset) {
        position = offset;
    }

    public int readInt() {
        try {
            return buffer.memory().getInt(position);
        } finally {
            position += Integer.BYTES;
        }
    }

    public void readFully(byte[] bytes) {
        read(bytes, 0, bytes.length);
    }

    private void read(byte[] bytes, int offset, int count) {
        try {
            buffer.memory().getBytes(position, bytes, offset, count);
        } finally {
            position += count;
        }
    }

    @Override
    public void readFully(float[] floats) {
        int bytesToRead = floats.length * Float.BYTES;
        if (floatsScratch.length < bytesToRead) {
            floatsScratch = new byte[bytesToRead];
        }
        read(floatsScratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(floatsScratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asFloatBuffer().get(floats);
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        int bytesToRead = (count - offset) * Integer.BYTES;
        if (intsScratch.length < bytesToRead) {
            intsScratch = new byte[bytesToRead];
        }
        read(intsScratch, 0, bytesToRead);
        ByteBuffer byteBuffer = ByteBuffer.wrap(intsScratch).order(ByteOrder.BIG_ENDIAN);
        byteBuffer.asIntBuffer().get(ints, offset, count);
    }

    @Override
    public void close() {
        // don't close buffer, let the Supplier handle that
    }
}
