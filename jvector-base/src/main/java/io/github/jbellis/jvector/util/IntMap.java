package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.NodesIterator;

public interface IntMap<T> {
    /**
     * @param key ordinal
     * @return true if successful, false if the current value != `existing`
     */
    boolean compareAndPut(int key, T existing, T value);

    /**
     * @return number of items that have been added
     */
    int size();

    /**
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    T get(int key);

    /**
     * @return the former value of the key, or null if it was not set
     */
    T remove(int key);

    /**
     * @return true iff the given key is set in the map
     */
    boolean containsKey(int key);

    /**
     * @return an iterator over all keys set in the map
     */
    NodesIterator keysIterator();

    /**
     * Iterates keys in ascending order and calls the consumer for each non-null key-value pair.
     */
    void forEach(IntBiConsumer<T> consumer);

    @FunctionalInterface
    interface IntBiConsumer<T2> {
        void consume(int key, T2 value);
    }
}
