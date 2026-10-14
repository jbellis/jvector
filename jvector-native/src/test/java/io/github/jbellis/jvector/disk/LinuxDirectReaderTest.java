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

import java.io.EOFException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

public class LinuxDirectReaderTest {
    private Path data() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        var path = Files.createTempFile("direct-reader", ".data");
        byte[] bytes = new byte[3 * 4096 + 137];
        new Random(51).nextBytes(bytes);
        Files.write(path, bytes);
        return path;
    }

    @Test
    public void testUnalignedValuesAndEndOfFile() throws Exception {
        Path path = data();
        try (var supplier = ReaderSupplierFactory.openDirect(path); var reader = supplier.get()) {
            byte[] bytes = Files.readAllBytes(path);
            var expected = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
            for (int offset : new int[] {0, 1, 4093, 4095, 4096, 8191, bytes.length - 8}) {
                reader.seek(offset);
                assertEquals(expected.getInt(offset), reader.readInt());
                reader.seek(offset);
                assertEquals(expected.getLong(offset), reader.readLong());
                reader.seek(offset);
                assertEquals(expected.getInt(offset), Float.floatToRawIntBits(reader.readFloat()));
            }
            for (int offset : new int[] {1, 4093, 4096, 8191}) {
                reader.seek(offset);
                byte[] actual = new byte[73];
                reader.readFully(actual);
                assertArrayEquals(Arrays.copyOfRange(bytes, offset, offset + actual.length), actual);
                var target = ByteBuffer.allocateDirect(73);
                target.position(3);
                target.limit(70);
                reader.seek(offset);
                reader.readFully(target);
                assertEquals(70, target.position());
                target.flip();
                target.position(3);
                byte[] copied = new byte[67];
                target.get(copied);
                assertArrayEquals(Arrays.copyOfRange(bytes, offset, offset + copied.length), copied);
                int[] ints = new int[1004];
                reader.seek(offset);
                reader.read(ints, 2, 1000);
                for (int n = 0; n < 1000; n++) assertEquals(expected.getInt(offset + n * 4), ints[n + 2]);
                float[] floats = new float[1004];
                reader.seek(offset);
                reader.read(floats, 2, 1000);
                for (int n = 0; n < 1000; n++)
                    assertEquals(expected.getInt(offset + n * 4), Float.floatToRawIntBits(floats[n + 2]));
                long[] longs = new long[500];
                reader.seek(offset);
                reader.readFully(longs);
                for (int n = 0; n < 500; n++) assertEquals(expected.getLong(offset + n * 8), longs[n]);
            }
            reader.seek(bytes.length - 1);
            assertThrows(EOFException.class, reader::readInt);
            assertEquals(bytes.length - 1, reader.getPosition());
            reader.seek(bytes.length);
            reader.readFully(new byte[0]);
            assertThrows(EOFException.class, () -> reader.readFully(new byte[1]));
            assertThrows(EOFException.class, () -> reader.seek(-1));
            assertThrows(EOFException.class, () -> reader.seek(bytes.length + 1));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testConcurrentCursorsAndClose() throws Exception {
        Path path = data();
        var executor = Executors.newFixedThreadPool(8);
        try (var supplier = new LinuxDirectReader.Supplier(path)) {
            var expected = ByteBuffer.wrap(Files.readAllBytes(path));
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int worker = 0; worker < 8; worker++) {
                final int seed = worker;
                tasks.add(executor.submit(() -> {
                    try (var reader = supplier.get()) {
                        var random = new Random(seed);
                        for (int n = 0; n < 1000; n++) {
                            int offset = random.nextInt(expected.capacity() - 8);
                            reader.seek(offset);
                            assertEquals(expected.getLong(offset), reader.readLong());
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }));
            }
            for (var task : tasks) task.get();
            var reader = supplier.get();
            reader.close();
            reader.close();
            assertThrows(java.io.IOException.class, reader::readInt);
            var outstanding = supplier.get();
            supplier.close();
            supplier.close();
            assertThrows(java.io.IOException.class, outstanding::readInt);
            assertThrows(java.io.IOException.class, supplier::get);
        } finally {
            executor.shutdown();
            assertTrue(executor.awaitTermination(30, TimeUnit.SECONDS));
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testPreparedRecordsAndBoundedFallback() throws Exception {
        assumeTrue(System.getProperty("os.name").equals("Linux"));
        var path = Files.createTempFile("direct-records", ".data");
        byte[] bytes = new byte[2 * 65536 + 137];
        new Random(17).nextBytes(bytes);
        Files.write(path, bytes);
        var expected = ByteBuffer.wrap(bytes);
        try (var supplier = new LinuxDirectReader.Supplier(path); var reader = supplier.get()) {
            // Read features then edges and return to features within one prepared record.
            for (int size : new int[] {4096, 8192, 16384, 65536}) {
                reader.seek(19);
                reader.prepareRead(4096, size);
                assertEquals(19, reader.getPosition());
                for (int offset : new int[] {4096 + size - 8, 4097, 8190, 4096}) {
                    reader.seek(offset);
                    assertEquals(expected.getLong(offset), reader.readLong());
                }
            }
            // Larger ranges retain ordinary read semantics without an unbounded buffer.
            reader.prepareRead(0, bytes.length);
            reader.seek(0);
            byte[] actual = new byte[bytes.length];
            reader.readFully(actual);
            assertArrayEquals(bytes, actual);
            reader.prepareRead(bytes.length - 20, 20);
            reader.seek(bytes.length - 20);
            byte[] tail = new byte[20];
            reader.readFully(tail);
            assertArrayEquals(Arrays.copyOfRange(bytes, bytes.length - 20, bytes.length), tail);
            reader.prepareRead(bytes.length, 0);
            assertThrows(EOFException.class, () -> reader.prepareRead(-1, 8));
            assertThrows(EOFException.class, () -> reader.prepareRead(0, -1));
            assertThrows(EOFException.class, () -> reader.prepareRead(bytes.length, 1));
            assertThrows(EOFException.class, () -> reader.prepareRead(1, Long.MAX_VALUE));
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testDescriptorReallyUsesDirectIO() throws Exception {
        Path path = data();
        try (var supplier = new LinuxDirectReader.Supplier(path)) {
            boolean found = false;
            try (var fds = Files.list(Path.of("/proc/self/fd"))) {
                for (var fd : fds.toList()) {
                    if (!Files.isSymbolicLink(fd) || !Files.readSymbolicLink(fd).equals(path.toAbsolutePath()))
                        continue;
                    var info = Files.readString(Path.of("/proc/self/fdinfo", fd.getFileName().toString()));
                    var flags = java.util.regex.Pattern.compile("(?m)^flags:\\s+([0-7]+)$").matcher(info);
                    assertTrue(flags.find());
                    assertTrue((Long.parseLong(flags.group(1), 8)
                                       & (System.getProperty("os.arch").equals("aarch64") ? 0200000 : 040000))
                            != 0); // Linux O_DIRECT.
                    found = true;
                }
            }
            assertTrue("Expected O_DIRECT descriptor", found);
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    public void testSupplierPinsOneFileGeneration() throws Exception {
        for (int depth : new int[] {0, 8}) {
            Path path = data();
            Path replacement = Files.createTempFile(path.getParent(), "replacement", ".data");
            byte[] original = Files.readAllBytes(path);
            byte[] changed = original.clone();
            Arrays.fill(changed, (byte) 77);
            Files.write(replacement, changed);
            try (var supplier = new LinuxDirectReader.Supplier(path, depth, 40000)) {
                try (var reader = supplier.get()) {
                    reader.seek(0);
                    assertEquals(ByteBuffer.wrap(original).getInt(), reader.readInt());
                }
                Files.move(replacement, path, java.nio.file.StandardCopyOption.ATOMIC_MOVE,
                        java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                // Existing suppliers retain their inode even when the pathname is replaced.
                try (var reader = supplier.get()) {
                    byte[] actual = new byte[original.length];
                    reader.readFully(actual);
                    assertArrayEquals(original, actual);
                }
                try (var fresh = new LinuxDirectReader.Supplier(path, depth); var reader = fresh.get()) {
                    byte[] actual = new byte[changed.length];
                    reader.readFully(actual);
                    assertArrayEquals(changed, actual);
                }
            } finally {
                Files.deleteIfExists(path);
                Files.deleteIfExists(replacement);
            }
        }
    }

    @Test
    public void testUncachedViewsDoNotPopulateSharedCache() throws Exception {
        for (int depth : new int[] {0, 8}) {
            Path path = data();
            try (var supplier = new LinuxDirectReader.Supplier(path, depth, 40000)) {
                try (var metadata = supplier.getUncached()) {
                    metadata.readInt();
                }
                long requests = supplier.statistics().ioRequests();
                assertEquals(1, requests);
                try (var reader = supplier.get()) {
                    reader.readInt();
                }
                assertEquals(requests + 1, supplier.statistics().ioRequests());
                try (var reader = supplier.get()) {
                    reader.readInt();
                }
                assertEquals(requests + 1, supplier.statistics().ioRequests());
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    @Test
    public void testUnusedSpeculativeReadsDoNotPopulateSharedCache() throws Exception {
        Path path = data();
        try (var supplier = new LinuxDirectReader.Supplier(path, 8, 40000); var speculative = supplier.get()) {
            speculative.prepareReads(new long[] {0, 4096}, new long[] {4096, 4096}, 2);
            long requests = supplier.statistics().ioRequests();
            assertEquals(2, requests);
            try (var reader = supplier.get()) {
                reader.readInt();
            }
            assertEquals(requests + 1, supplier.statistics().ioRequests());
            // Consume a prepared block without another physical read.
            speculative.seek(4096);
            speculative.readInt();
            assertEquals(requests + 1, supplier.statistics().ioRequests());
            try (var reader = supplier.get()) {
                reader.seek(4096);
                reader.readInt();
            }
            assertEquals(requests + 1, supplier.statistics().ioRequests());
        } finally {
            Files.deleteIfExists(path);
        }
    }
}
