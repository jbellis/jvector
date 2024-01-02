/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.graph.NodesIterator;

import java.util.AbstractMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.StampedLock;
import java.util.stream.IntStream;

/**
 * A map (but not a Map) of int -> T where the int keys are dense-ish and start at zero,
 * but the size of the map is not known in advance.  This provides fast, concurrent
 * updates and minimizes contention when the map is resized.
 * <p>
 * "Dense-ish" means that space is allocated for all keys from 0 to the highest key, but
 * it is valid to have gaps in the keys.  The value associated with "gap" keys is null.
 */
public class DenseIntMap<T> {
    private volatile AtomicReferenceArray<T> objects;
    private final AtomicInteger size;
    private final StampedLock sl = new StampedLock();

    public DenseIntMap(int initialSize) {
        objects = new AtomicReferenceArray<>(initialSize);
        size = new AtomicInteger();
    }

    /**
     * @param key ordinal
     */
    public void put(int key, T value) {
        if (value == null) {
            throw new IllegalArgumentException("put() value cannot be null -- use remove() instead");
        }

        ensureCapacity(key);
        long stamp;
        boolean isInsert = false;
        do {
            stamp = sl.tryOptimisticRead();
            isInsert = objects.getAndSet(key, value) == null || isInsert;
        } while (!sl.validate(stamp));

        if (isInsert) {
            size.incrementAndGet();
        }
    }

    /**
     * @return number of items that have been added
     */
    public int size() {
        return size.get();
    }

    /**
     * @param key ordinal
     * @return the value of the key, or null if not set
     */
    public T get(int key) {
        // since objects is volatile, we don't need to lock
        var ref = objects;
        if (key >= ref.length()) {
            return null;
        }
        return ref.get(key);
    }

    private void ensureCapacity(int node) {
        if (node < objects.length()) {
            return;
        }

        long stamp = sl.writeLock();
        try {
            var oldArray = objects;
            if (node >= oldArray.length()) {
                int newSize = ArrayUtil.oversize(node + 1, RamUsageEstimator.NUM_BYTES_OBJECT_REF);
                var newArray = new AtomicReferenceArray<T>(newSize);
                for (int i = 0; i < oldArray.length(); i++) {
                    newArray.set(i, oldArray.get(i));
                }
                objects = newArray;
            }
        } finally {
            sl.unlockWrite(stamp);
        }
    }

    /**
     * @return the former value of the key, or null if it was not set
     */
    public T remove(int key) {
        if (key >= objects.length()) {
            return null;
        }
        var old = objects.get(key);
        if (old == null) {
            return null;
        }

        // ensureCapacity doesn't spin, so we respect the Big Lock to make sure `objects` doesn't
        // change out from underneath us
        long stamp = sl.writeLock();
        try {
            if (objects.compareAndSet(key, old, null)) {
                size.decrementAndGet();
                return old;
            } else {
                return null;
            }
        } finally {
            sl.unlockWrite(stamp);
        }
    }

    public boolean containsKey(int key) {
        return get(key) != null;
    }

    public Set<Map.Entry<Integer, T>> entrySet() {
        var entries = new HashSet<Map.Entry<Integer, T>>(size());
        var ref = objects;
        for (int i = 0; i < ref.length(); i++) {
            var value = ref.get(i);
            if (value != null) {
                entries.add(new AbstractMap.SimpleEntry<>(i, value));
            }
        }
        return entries;
    }

    public Set<Integer> keySet() {
        var keys = new HashSet<Integer>(size());
        var ref = objects;
        for (int i = 0; i < ref.length(); i++) {
            var value = ref.get(i);
            if (value != null) {
                keys.add(i);
            }
        }
        return keys;
    }

    public NodesIterator getNodesIterator() {
        // implemented here because we can't make it threadsafe AND performant elsewhere
        var minSize = size(); // if keys are added concurrently we will miss them
        var ref = objects;
        var keysInts = IntStream.range(0, ref.length()).filter(i -> ref.get(i) != null).iterator();
        return NodesIterator.fromPrimitiveIterator(keysInts, minSize);
    }
}
