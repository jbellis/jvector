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

package io.github.jbellis.jvector.disk;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.locks.ReentrantLock;

/** Fixed-size four-way sector cache. Locks protect copies as well as slot replacement. */
final class DirectBlockCache implements AutoCloseable {
    private static final int BLOCK = 4096, WAYS = 4, FIXED_ALLOWANCE = 16384;
    private final Arena arena;
    private final MemorySegment data;
    private final long[] tags, ages, clocks;
    private final int[] valid;
    private final ReentrantLock[] locks;
    private final int sets;
    private volatile boolean closed;
    final long reservedBytes;
    final java.util.concurrent.atomic.LongAdder hits = new java.util.concurrent.atomic.LongAdder();
    final java.util.concurrent.atomic.LongAdder misses = new java.util.concurrent.atomic.LongAdder();
    final java.util.concurrent.atomic.LongAdder evictions = new java.util.concurrent.atomic.LongAdder();

    DirectBlockCache(long budget) {
        if (budget < FIXED_ALLOWANCE + (BLOCK + 32L) * WAYS
                || (budget - FIXED_ALLOWANCE) / ((BLOCK + 32L) * WAYS) > Integer.MAX_VALUE / WAYS)
            throw new IllegalArgumentException(
                    "Cache budget must be at least 32896 bytes and fit the cache slot index");
        sets = (int) ((budget - FIXED_ALLOWANCE) / ((BLOCK + 32L) * WAYS));
        int slots = sets * WAYS;
        reservedBytes = FIXED_ALLOWANCE + slots * (BLOCK + 32L); // Payload plus conservative metadata allowance.
        tags = new long[slots];
        ages = new long[slots];
        valid = new int[slots];
        Arrays.fill(tags, -1);
        locks = new ReentrantLock[Integer.highestOneBit(Math.min(16, sets))];
        clocks = new long[locks.length];
        for (int i = 0; i < locks.length; i++) locks[i] = new ReentrantLock();
        arena = Arena.ofShared();
        try {
            data = arena.allocate(slots * (long) BLOCK, BLOCK);
        } catch (Throwable t) {
            arena.close();
            throw t;
        }
    }

    private int set(long offset) {
        long h = offset >>> 12;
        h = (h ^ (h >>> 33)) * 0xff51afd7ed558ccdL;
        h = (h ^ (h >>> 33)) * 0xc4ceb9fe1a85ec53L;
        // Scale an unsigned 32-bit hash to any set count without division.
        return (int) (((h ^ (h >>> 33)) & 0xffffffffL) * sets >>> 32);
    }

    private int find(int base, long block, int needed) {
        for (int i = base; i < base + WAYS; i++)
            if (tags[i] == block && valid[i] >= needed)
                return i;
        return -1;
    }

    boolean contains(long offset, long length) {
        if (length < 0 || closed)
            return false;
        while (length > 0) {
            long block = offset & -4096L;
            int delta = (int) (offset - block), n = (int) Math.min(length, BLOCK - delta);
            int set = set(block), stripe = set & (locks.length - 1);
            locks[stripe].lock();
            try {
                if (closed || find(set * WAYS, block, delta + n) < 0)
                    return false;
            } finally {
                locks[stripe].unlock();
            }
            offset += n;
            length -= n;
        }
        return true;
    }

    boolean copy(long offset, int length, MemorySegment target) {
        int done = 0;
        while (done < length) {
            long block = offset & -4096L;
            int delta = (int) (offset - block), n = Math.min(length - done, BLOCK - delta);
            int set = set(block), stripe = set & (locks.length - 1);
            locks[stripe].lock();
            try {
                int slot = find(set * WAYS, block, delta + n);
                if (closed || slot < 0) {
                    misses.increment();
                    return false;
                }
                MemorySegment.copy(data, slot * (long) BLOCK + delta, target, done, n);
                ages[slot] = ++clocks[stripe];
            } finally {
                locks[stripe].unlock();
            }
            offset += n;
            done += n;
        }
        hits.increment();
        return true;
    }

    void put(long offset, MemorySegment source, int length) {
        // Callers publish completed, block-aligned immutable reads only.
        if ((offset & 4095) != 0)
            throw new IllegalArgumentException("Unaligned cache publication");
        int done = 0;
        while (done < length) {
            long block = offset + done;
            int n = Math.min(BLOCK, length - done);
            int set = set(block), stripe = set & (locks.length - 1), base = set * WAYS;
            locks[stripe].lock();
            try {
                if (closed)
                    return;
                int slot = find(base, block, 0);
                if (slot >= 0 && valid[slot] >= n) {
                    done += n;
                    continue;
                }
                if (slot < 0) {
                    slot = base;
                    for (int i = base; i < base + WAYS; i++) {
                        if (tags[i] < 0) {
                            slot = i;
                            break;
                        }
                        if (ages[i] < ages[slot])
                            slot = i;
                    }
                }
                if (tags[slot] >= 0 && tags[slot] != block)
                    evictions.increment();
                MemorySegment.copy(source, done, data, slot * (long) BLOCK, n);
                tags[slot] = block;
                valid[slot] = n;
                ages[slot] = ++clocks[stripe];
            } finally {
                locks[stripe].unlock();
            }
            done += n;
        }
    }

    @Override
    public void close() {
        for (var lock : locks) lock.lock();
        try {
            if (!closed) {
                closed = true;
                arena.close();
            }
        } finally {
            for (int i = locks.length - 1; i >= 0; i--) locks[i].unlock();
        }
    }
}
