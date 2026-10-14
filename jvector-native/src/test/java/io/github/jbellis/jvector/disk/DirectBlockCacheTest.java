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

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

public class DirectBlockCacheTest {
    @Test
    public void testBudgetReplacementAndPartialTail() throws Exception {
        try (var cache = new DirectBlockCache(40000); var arena = Arena.ofConfined()) {
            assertTrue(cache.reservedBytes <= 40000);
            var block = arena.allocate(4096, 4096);
            var target = arena.allocate(4096, 4096);
            for (int n = 0; n < 4; n++) {
                block.fill((byte) n);
                cache.put(n * 4096L, block, 4096);
            }
            assertTrue(cache.copy(0, 4096, target));
            block.fill((byte) 4);
            cache.put(4 * 4096L, block, 137);
            assertTrue(cache.contains(0, 4096));
            assertFalse(cache.contains(4096, 1));
            assertTrue(cache.copy(4 * 4096L + 10, 127, target));
            assertEquals(4, target.get(ValueLayout.JAVA_BYTE, 126));
            assertFalse(cache.contains(4 * 4096L, 138));
            assertEquals(1, cache.evictions.sum());
            cache.close();
            assertFalse(cache.copy(0, 1, target));
        }
        assertThrows(IllegalArgumentException.class, () -> new DirectBlockCache(4096));
        assertThrows(IllegalArgumentException.class, () -> new DirectBlockCache(Long.MAX_VALUE));
    }

    @Test
    public void testArbitraryBudgetUsesAllButOneSet() {
        try (var cache = new DirectBlockCache(1024 * 1024)) {
            assertTrue(cache.reservedBytes <= 1024 * 1024);
            assertTrue(1024 * 1024 - cache.reservedBytes < 16512);
        }
    }

    @Test
    public void testSharedCacheSkipsIOAcrossViews() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        var path = Files.createTempFile("cache-views", ".data");
        byte[] bytes = new byte[8192];
        new java.util.Random(9).nextBytes(bytes);
        Files.write(path, bytes);
        try (var supplier = new LinuxDirectReader.Supplier(path, 8, 40000)) {
            try (var reader = supplier.get()) {
                reader.prepareRead(0, 8192);
                reader.seek(0);
                byte[] actual = new byte[8192];
                reader.readFully(actual);
                assertArrayEquals(bytes, actual);
            }
            long requests = supplier.statistics().ioRequests();
            assertEquals(1, requests);
            try (var reader = supplier.get()) {
                reader.prepareReads(new long[] {0, 4096}, new long[] {4096, 4096}, 2);
                reader.seek(0);
                byte[] actual = new byte[8192];
                reader.readFully(actual);
                assertArrayEquals(bytes, actual);
            }
            assertEquals(requests, supplier.statistics().ioRequests());
            assertTrue(supplier.statistics().cacheHits() > 0);
            assertTrue(supplier.statistics().cacheReservedBytes() <= 40000);
            assertEquals(0, supplier.statistics().viewBufferBytes());
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testConcurrentEvictionDoesNotExposeMixedBytes() throws Exception {
        var pool = Executors.newFixedThreadPool(8);
        try (var cache = new DirectBlockCache(40000)) {
            var jobs = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int worker = 0; worker < 8; worker++) {
                final int seed = worker;
                jobs.add(pool.submit(() -> {
                    try (var arena = Arena.ofConfined()) {
                        var source = arena.allocate(4096, 4096);
                        var target = arena.allocate(4096, 4096);
                        var random = new java.util.Random(seed);
                        for (int i = 0; i < 2000; i++) {
                            int key = random.nextInt(16);
                            source.fill((byte) key);
                            cache.put(key * 4096L, source, 4096);
                            if (cache.copy(key * 4096L, 4096, target)) {
                                byte[] expected = new byte[4096];
                                Arrays.fill(expected, (byte) key);
                                assertArrayEquals(expected, target.toArray(ValueLayout.JAVA_BYTE));
                            }
                        }
                    }
                }));
            }
            for (var job : jobs) job.get();
        } finally {
            pool.shutdown();
            assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
        }
    }
}
