package io.github.jbellis.jvector.example.util;

import java.io.Closeable;
import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

public class MMapRandomAccessVectorValues implements RandomAccessVectorValues<float[]>, Closeable {
    final int dimension;
    final int rows;
    final File file;
    final float[] valueBuffer;

    final MMapBuffer fileReader;

    public MMapRandomAccessVectorValues(File f, int dimension) {
        assert f != null && f.exists() && f.canRead();
        assert f.length() % ((long) dimension * Float.BYTES) == 0;

        try {
            this.file = f;
            this.fileReader = new MMapBuffer(f, FileChannel.MapMode.READ_ONLY, ByteOrder.LITTLE_ENDIAN);
            this.dimension = dimension;
            this.rows = ((int) f.length()) / dimension;
            this.valueBuffer = new float[dimension];
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    @Override
    public int size() {
        return (int) (file.length() / ((long) dimension * Float.BYTES));
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        long offset = (long) targetOrd * dimension * Float.BYTES;
        int i = 0;
        for (long o = offset; o < offset + ((long) dimension * Float.BYTES); o += Float.BYTES, i++)
            valueBuffer[i] = fileReader.memory().getFloat(o);

        return valueBuffer;
    }

    @Override
    public boolean isValueShared() {
        return true;
    }

    @Override
    public RandomAccessVectorValues<float[]> copy() {
        return new MMapRandomAccessVectorValues(file, dimension);
    }

    @Override
    public void close() {
        try {
            this.fileReader.close();
        } catch (IOException e) {
            throw new IOError(e);
        }
    }
}
