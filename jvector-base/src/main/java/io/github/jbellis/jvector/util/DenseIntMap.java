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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.StampedLock;

/**
 * A Map of int -> T where the int keys are dense and start at zero, but the
 * size of the map is not known in advance.  This provides fast, concurrent
 * updates and minimizes contention when the map is resized.
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
        ensureCapacity(key);
        long stamp;
        do {
            stamp = sl.tryOptimisticRead();
            objects.set(key, value);
        } while (!sl.validate(stamp));

        size.incrementAndGet();
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
}