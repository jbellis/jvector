package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;

public interface RandomAccessWriter extends AutoCloseable {
    void seek(long position) throws IOException;

    void writeFully(VectorFloat<?> vector) throws IOException;
}
