package com.github.jbellis.jvector.disk;

public interface ReaderSupplier extends AutoCloseable {
    public RandomAccessReader get();
}
