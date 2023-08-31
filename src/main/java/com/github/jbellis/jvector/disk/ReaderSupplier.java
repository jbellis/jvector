package com.github.jbellis.jvector.disk;

public interface ReaderSupplier extends AutoCloseable {
    RandomAccessReader get();

    default void close() {
    }
}
