package io.github.jbellis.jvector.example.util;

import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOError;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;

public class MMapRandomAccessVectorValues implements RandomAccessVectorValues<float[]>, Closeable
{

    final int dimension;
    final int rows;
    final File file;
    final byte[] byteBuffer;
    final float[] valueBuffer;

    final RandomAccessFile fileReader;

    public MMapRandomAccessVectorValues(File f, int dimension) {
        assert f != null && f.exists() && f.canRead();
        assert f.length() % ((long) dimension * Float.BYTES) == 0;

        try {
            this.file = f;
            this.fileReader = new RandomAccessFile(f, "r");
            this.dimension = dimension;
            this.rows = ((int) f.length()) / dimension;
            this.byteBuffer = new byte[dimension * Float.BYTES];
            this.valueBuffer = new float[dimension];
        } catch (FileNotFoundException e) {
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
        try {
            fileReader.seek((long) targetOrd * dimension * Float.BYTES);
            fileReader.readFully(byteBuffer);
            ByteBuffer.wrap(byteBuffer).asFloatBuffer().get(valueBuffer);
            return valueBuffer;
        } catch (IOException e) {
            throw new IOError(e);
        }
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
