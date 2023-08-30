package com.github.jbellis.jvector.disk;

import java.io.DataInput;
import java.io.IOException;

public interface RandomAccessReader extends DataInput, AutoCloseable {
    public void seek(long offset) throws IOException;
}
