package com.github.jbellis.jvector.disk;

import java.io.IOException;

/**
 * This is a subset of DataInput, plus seek and readFully(float[]), which allows implementations
 * to use a more efficient option like FloatBuffer.
 */
public interface RandomAccessReader extends AutoCloseable {
    public void seek(long offset) throws IOException;

    public int readInt() throws IOException;

    public void readFully(byte[] bytes) throws IOException;

    public void readFully(float[] floats) throws IOException;
}
