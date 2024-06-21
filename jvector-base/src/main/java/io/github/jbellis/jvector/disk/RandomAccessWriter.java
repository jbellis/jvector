package io.github.jbellis.jvector.disk;

import java.io.Closeable;
import java.io.DataOutput;
import java.io.IOException;

/**
 * A DataOutput that adds methods for random access writes
 */
public interface RandomAccessWriter extends DataOutput, Closeable {
    void seek(long position) throws IOException;

    long position() throws IOException;

    void flush() throws IOException;

    long checksum(long startOffset, long endOffset) throws IOException;
}
