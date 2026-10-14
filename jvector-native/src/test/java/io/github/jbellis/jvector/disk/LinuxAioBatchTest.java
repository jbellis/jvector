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
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

public class LinuxAioBatchTest {
    @Test
    public void testRealBatchDeduplicationAndReuse() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        Path path = Files.createTempFile("aio-batch", ".data");
        byte[] expected = new byte[3 * 4096 + 137];
        new Random(17).nextBytes(expected);
        Files.write(path, expected);
        try (var io = new LinuxAioBatch(path, 4); var arena = Arena.ofConfined()) {
            long[] offsets = {0, 4096, 0, 12288}, lengths = {4096, 4096, 4096, 137};
            io.prepare(offsets, lengths, 4);
            assertEquals(3, io.readRequests);
            assertEquals(1, io.submissions);
            // Verify the descriptor used by kernel AIO itself, not only the JDK reader.
            int directFlag = System.getProperty("os.arch").equals("aarch64") ? 0x10000 : 0x4000;
            int found = 0;
            try (var fds = Files.list(Path.of("/proc/self/fd"))) {
                for (var fd : fds.toList()) {
                    if (!Files.isSymbolicLink(fd) || !Files.readSymbolicLink(fd).equals(path.toAbsolutePath()))
                        continue;
                    var info = Files.readString(Path.of("/proc/self/fdinfo", fd.getFileName().toString()));
                    var flags = java.util.regex.Pattern.compile("(?m)^flags:\\s+([0-7]+)$").matcher(info);
                    assertTrue(flags.find());
                    assertTrue((Long.parseLong(flags.group(1), 8) & directFlag) != 0);
                    found++;
                }
            }
            assertEquals(1, found);

            var target = arena.allocate(4096, 4096);
            for (int i = 0; i < 4; i++) {
                assertTrue(io.copy(offsets[i], (int) lengths[i], target));
                byte[] actual = target.asSlice(0, lengths[i]).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE);
                assertArrayEquals(
                        java.util.Arrays.copyOfRange(expected, (int) offsets[i], (int) (offsets[i] + lengths[i])),
                        actual);
            }
            io.prepare(offsets, lengths, 4);
            assertEquals(3, io.readRequests);
            io.close();
            assertThrows(java.io.IOException.class, () -> io.prepare(offsets, lengths, 4));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testBatchReaderConcurrentViews() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        Path path = Files.createTempFile("aio-views", ".data");
        byte[] bytes = new byte[65 * 4096 + 137];
        new Random(51).nextBytes(bytes);
        Files.write(path, bytes);
        var pool = Executors.newFixedThreadPool(8);
        try (var supplier = ReaderSupplierFactory.openDirect(path, 8)) {
            var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int worker = 0; worker < 8; worker++) {
                final int seed = worker;
                futures.add(pool.submit(() -> {
                    try (var reader = supplier.get()) {
                        var random = new Random(seed);
                        var expected = ByteBuffer.wrap(bytes);
                        var offsets = new long[8];
                        var lengths = new long[8];
                        for (int batch = 0; batch < 25; batch++) {
                            for (int i = 0; i < 8; i++) {
                                offsets[i] = 4096L * random.nextInt(65);
                                lengths[i] = 4096;
                            }
                            reader.prepareReads(offsets, lengths, 8);
                            for (int i = 0; i < 8; i++) {
                                int pos = (int) offsets[i] + random.nextInt(4088);
                                reader.seek(pos);
                                assertEquals(expected.getLong(pos), reader.readLong());
                            }
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }));
            }
            for (var future : futures) future.get();
        } finally {
            pool.shutdown();
            assertTrue(pool.awaitTermination(30, TimeUnit.SECONDS));
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testReadFailureDrainsBeforeClose() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        Path path = Files.createTempFile("aio-short", ".data");
        Files.write(path, new byte[8192]);
        try (var io = new LinuxAioBatch(path, 4)) {
            Files.write(path, new byte[0]); // Violating immutability must fail safely, not expose old bytes.
            assertThrows(java.io.IOException.class, () -> io.prepare(new long[] {0, 4096}, new long[] {4096, 4096}, 2));
            io.close();
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
