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

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

public class TestDenseIntMapLocking {
    @Test
    public void testInsertRemoveRaceKeepsSizeConsistent() throws Exception {
        // Regression for #218: a concurrent insertion/removal must not leave size
        // permanently different from the entries present after both operations finish.
        DenseIntMap<Object> map = new DenseIntMap<>(1);
        Object value = new Object();
        int rounds = 20_000;
        CyclicBarrier phase = new CyclicBarrier(3);
        ExecutorService workers = Executors.newFixedThreadPool(2);
        try {
            Future<?> insert = workers.submit(() -> {
                for (int round = 0; round < rounds; round++) {
                    phase.await(20, TimeUnit.SECONDS);
                    map.compareAndPut(0, null, value);
                    phase.await(20, TimeUnit.SECONDS);
                }
                return null;
            });
            Future<?> remove = workers.submit(() -> {
                for (int round = 0; round < rounds; round++) {
                    phase.await(20, TimeUnit.SECONDS);
                    map.remove(0);
                    phase.await(20, TimeUnit.SECONDS);
                }
                return null;
            });
            for (int round = 0; round < rounds; round++) {
                phase.await(20, TimeUnit.SECONDS);
                phase.await(20, TimeUnit.SECONDS);
                // Workers cannot mutate again until the next phase starts.
                assertEquals("size after insert/remove race " + round,
                        map.get(0) == null ? 0 : 1, map.size());
                map.remove(0);
                assertEquals(0, map.size());
            }
            insert.get(20, TimeUnit.SECONDS);
            remove.get(20, TimeUnit.SECONDS);
        } finally {
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(25, TimeUnit.SECONDS));
        }
    }

    @Test
    public void testUpdatesAndRemovalsDuringResize() throws Exception {
        int threads = 48;
        int keysPerThread = 1_024;
        DenseIntMap<Integer> map = new DenseIntMap<>(1);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService workers = Executors.newFixedThreadPool(threads);
        List<Future<?>> tasks = new ArrayList<>();
        try {
            for (int worker = 0; worker < threads; worker++) {
                int ordinal = worker;
                tasks.add(workers.submit(() -> {
                    start.await();
                    for (int i = 0; i < keysPerThread; i++) {
                        int key = i * threads + ordinal;
                        Integer original = key;
                        assertTrue(map.compareAndPut(key, null, original));
                        assertFalse(map.compareAndPut(key, null, -1));
                        if ((key & 1) == 0) {
                            assertSame(original, map.remove(key));
                        } else {
                            assertTrue(map.compareAndPut(key, original, key + 1));
                            assertFalse(map.compareAndPut(key, original, -1));
                        }
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> task : tasks) {
                task.get(30, TimeUnit.SECONDS);
            }
            assertEquals(threads * keysPerThread / 2, map.size());
            for (int key = 0; key < threads * keysPerThread; key++) {
                if ((key & 1) == 0) {
                    assertNull(map.get(key));
                } else {
                    assertEquals(Integer.valueOf(key + 1), map.get(key));
                }
            }
            map.forEach((key, value) -> assertEquals(Integer.valueOf(key + 1), value));
        } finally {
            start.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(25, TimeUnit.SECONDS));
        }
    }

    @Test
    public void testContendedInsertCountsOnlyOneWinner() throws Exception {
        int threads = 48;
        DenseIntMap<Object> map = new DenseIntMap<>(1);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService workers = Executors.newFixedThreadPool(threads);
        List<Future<Boolean>> tasks = new ArrayList<>();
        try {
            for (int worker = 0; worker < threads; worker++) {
                tasks.add(workers.submit(() -> {
                    start.await();
                    return map.compareAndPut(0, null, new Object());
                }));
            }
            start.countDown();
            int winners = 0;
            for (Future<Boolean> task : tasks) {
                if (task.get(30, TimeUnit.SECONDS)) {
                    winners++;
                }
            }
            assertEquals(1, winners);
            assertEquals(1, map.size());
            assertNotNull(map.remove(0));
            assertEquals(0, map.size());
        } finally {
            start.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(25, TimeUnit.SECONDS));
        }
    }
}
