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

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.concurrent.atomic.LongAdder;

/** One pinned immutable file description, shared by positional readers and AIO contexts. */
final class LinuxDirectFile implements AutoCloseable {
    private static final Linker LINKER = Linker.nativeLinker();
    static final MemoryLayout ERRNO = Linker.Option.captureStateLayout();
    static final long ERRNO_OFFSET = ERRNO.byteOffset(MemoryLayout.PathElement.groupElement("errno"));
    private static final MethodHandle OPEN =
            function("open", ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT);
    private static final MethodHandle PREAD = function("pread", ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG);
    private static final MethodHandle LSEEK =
            function("lseek", ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT);
    private static final MethodHandle CLOSE = function("close", ValueLayout.JAVA_INT, ValueLayout.JAVA_INT);
    private static final MethodHandle STATX =
            LINKER.defaultLookup()
                    .find("statx")
                    .map(symbol
                            -> LINKER.downcallHandle(symbol,
                                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                            ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                                            ValueLayout.ADDRESS),
                                    Linker.Option.captureCallState("errno")))
                    .orElse(null);
    private volatile int fd = -1;
    final long size;
    final LongAdder synchronousRequests = new LongAdder();

    private static MethodHandle function(
            String name, java.lang.foreign.MemoryLayout result, java.lang.foreign.MemoryLayout... arguments) {
        return LINKER.downcallHandle(LINKER.defaultLookup().find(name).orElseThrow(),
                FunctionDescriptor.of(result, arguments), Linker.Option.captureCallState("errno"));
    }

    LinuxDirectFile(Path path) throws IOException {
        String arch = System.getProperty("os.arch");
        if (!System.getProperty("os.name").equals("Linux") || ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN
                || !(arch.equals("amd64") || arch.equals("x86_64") || arch.equals("aarch64"))) {
            throw new IOException("Direct I/O requires little-endian x86_64 or AArch64 Linux");
        }
        int direct = arch.equals("aarch64") ? 0x10000 : 0x4000;
        try (var arena = Arena.ofConfined()) {
            var state = arena.allocate(ERRNO);
            fd = (int) OPEN.invokeExact(state, arena.allocateFrom(path.toString()), direct | 0x80000);
            if (fd < 0)
                throw failure("open O_DIRECT", state);
            long length = (long) LSEEK.invokeExact(state, fd, 0L, 2); // SEEK_END; reads are positional.
            if (length < 0)
                throw failure("lseek", state);
            checkAlignment(arena, state);
            size = length;
        } catch (Throwable t) {
            try {
                close();
            } catch (IOException e) {
                t.addSuppressed(e);
            }
            if (t instanceof IOException e)
                throw e;
            throw new IOException("Cannot open direct-I/O file", t);
        }
    }

    private void checkAlignment(Arena arena, MemorySegment state) throws Throwable {
        if (STATX == null)
            return;
        // Linux statx is a fixed 256-byte ABI; query this descriptor, not its replaceable pathname.
        var result = arena.allocate(256, 8);
        result.fill((byte) 0);
        int rc = (int) STATX.invokeExact(state, fd, arena.allocateFrom(""), 0x1000, 0x2000, result);
        if (rc != 0 || (result.get(ValueLayout.JAVA_INT, 0) & 0x2000) == 0)
            return;
        int memoryAlignment = result.get(ValueLayout.JAVA_INT, 152);
        int offsetAlignment = result.get(ValueLayout.JAVA_INT, 156);
        if (memoryAlignment == 0 || offsetAlignment == 0 || 4096 % memoryAlignment != 0
                || 4096 % offsetAlignment != 0) {
            throw new IOException(
                    "Unsupported direct-I/O alignment: memory=" + memoryAlignment + ", offset=" + offsetAlignment);
        }
    }

    boolean isOpen() {
        return fd >= 0;
    }

    int descriptor() throws IOException {
        int descriptor = fd;
        if (descriptor < 0)
            throw new IOException("Direct file is closed");
        return descriptor;
    }

    int read(MemorySegment state, MemorySegment destination, int bytes, long offset) throws IOException {
        try {
            long n;
            do {
                n = (long) PREAD.invokeExact(state, descriptor(), destination, (long) bytes, offset);
            } while (n < 0 && state.get(ValueLayout.JAVA_INT, ERRNO_OFFSET) == 4); // EINTR.
            if (n < 0)
                throw failure("pread", state);
            synchronousRequests.increment();
            return Math.toIntExact(n);
        } catch (Throwable t) {
            if (t instanceof IOException e)
                throw e;
            throw new IOException("Direct read failed", t);
        }
    }

    private static IOException failure(String operation, MemorySegment state) {
        return new IOException(operation + " failed: errno=" + state.get(ValueLayout.JAVA_INT, ERRNO_OFFSET));
    }

    @Override
    public synchronized void close() throws IOException {
        int descriptor = fd;
        if (descriptor < 0)
            return;
        fd = -1;
        try (var arena = Arena.ofConfined()) {
            var state = arena.allocate(ERRNO);
            int rc = (int) CLOSE.invokeExact(state, descriptor);
            // Linux releases the descriptor even on EINTR; retrying could close a reused descriptor.
            if (rc < 0 && state.get(ValueLayout.JAVA_INT, ERRNO_OFFSET) != 4)
                throw failure("close", state);
        } catch (Throwable t) {
            if (t instanceof IOException e)
                throw e;
            throw new IOException("Cannot close direct file", t);
        }
    }
}
