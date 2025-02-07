package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.NodesIterator;

public interface IntMap<T> {
    boolean compareAndPut(int key, T existing, T value);

    int size();

    T get(int key);

    T remove(int key);

    boolean containsKey(int key);

    NodesIterator keysIterator();

    void forEach(IntBiConsumer<T> consumer);

    @FunctionalInterface
    interface IntBiConsumer<T2> {
        void consume(int key, T2 value);
    }
}
