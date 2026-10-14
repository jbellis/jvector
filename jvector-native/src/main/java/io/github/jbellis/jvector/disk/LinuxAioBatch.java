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
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;

/** Linux kernel AIO, with all submitted buffers kept alive until completion or io_destroy. */
final class LinuxAioBatch implements AutoCloseable {
    static final int MAX_RECORD_BYTES = 64 * 1024;
    private static final Linker LINKER = Linker.nativeLinker();
    private static final StructLayout ERRNO = Linker.Option.captureStateLayout();
    private static final long ERRNO_OFFSET = ERRNO.byteOffset(MemoryLayout.PathElement.groupElement("errno"));
    private static final MethodHandle SYSCALL =
            LINKER.downcallHandle(LINKER.defaultLookup().find("syscall").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG),
                    Linker.Option.firstVariadicArg(1), Linker.Option.captureCallState("errno"));
    private final Arena arena;
    private final MemorySegment errno;
    private final MemorySegment control, pointers, events, contextAddress;
    private final MemorySegment[] buffers;
    private final long[] starts;
    private final int[] valid, pendingSlots, pendingBytes;
    private final boolean[] pinned;
    private final long setup, destroy, submit, getevents;
    private final long fileSize;
    private final DirectBlockCache cache;
    private final int fd;
    private final LinuxDirectFile ownedFile;
    private long context;
    private boolean closed;
    long readRequests, submissions;

    LinuxAioBatch(Path path, int depth) throws IOException {
        this(openFile(path, depth), depth, null, true);
    }

    private static LinuxDirectFile openFile(Path path, int depth) throws IOException {
        if (depth < 1 || depth > 32)
            throw new IllegalArgumentException("AIO depth must be 1..32");
        return new LinuxDirectFile(path);
    }

    LinuxAioBatch(LinuxDirectFile file, int depth, DirectBlockCache cache) throws IOException {
        this(file, depth, cache, false);
    }

    private LinuxAioBatch(LinuxDirectFile file, int depth, DirectBlockCache cache, boolean ownsFile)
            throws IOException {
        if (depth < 1 || depth > 32)
            throw new IllegalArgumentException("AIO depth must be 1..32");
        this.cache = cache;
        this.ownedFile = ownsFile ? file : null;
        this.fd = file.descriptor();
        if (!System.getProperty("os.name").equals("Linux") || ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN)
            throw new IOException("Kernel AIO requires supported little-endian Linux");
        String arch = System.getProperty("os.arch");
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            setup = 206;
            destroy = 207;
            submit = 209;
            getevents = 208;
        } else if (arch.equals("aarch64")) {
            setup = 0;
            destroy = 1;
            submit = 2;
            getevents = 4;
        } else
            throw new IOException("Unsupported Linux AIO ABI: " + arch);
        this.fileSize = file.size;
        arena = Arena.ofShared();
        try {
            errno = arena.allocate(ERRNO);
            control = arena.allocate(depth * 64L, 8);
            pointers = arena.allocate(depth * 8L, 8);
            events = arena.allocate(depth * 32L, 8);
            contextAddress = arena.allocate(ValueLayout.JAVA_LONG);
            buffers = new MemorySegment[depth];
            starts = new long[depth];
            valid = new int[depth];
            pendingSlots = new int[depth];
            pendingBytes = new int[depth];
            pinned = new boolean[depth];
            Arrays.fill(starts, -1);
            for (int i = 0; i < depth; i++) buffers[i] = arena.allocate(MAX_RECORD_BYTES, 4096);
            call(setup, depth, contextAddress.address(), 0, 0, 0);
            context = contextAddress.get(ValueLayout.JAVA_LONG, 0);
        } catch (Throwable t) {
            try {
                close();
            } catch (IOException e) {
                t.addSuppressed(e);
            }
            if (t instanceof IOException e)
                throw e;
            throw new IOException("Cannot initialize Linux AIO", t);
        }
    }

    private IOException failure(String operation) {
        return new IOException(operation + " failed: errno=" + errno.get(ValueLayout.JAVA_INT, ERRNO_OFFSET));
    }

    private long raw(long number, long a, long b, long c, long d, long e) throws IOException {
        try {
            return (long) SYSCALL.invokeExact(errno, number, a, b, c, d, e);
        } catch (Throwable t) {
            throw new IOException("Linux AIO downcall failed", t);
        }
    }

    private long call(long number, long a, long b, long c, long d, long e) throws IOException {
        long result;
        do {
            result = raw(number, a, b, c, d, e);
        } while (result < 0 && errno.get(ValueLayout.JAVA_INT, ERRNO_OFFSET) == 4); // EINTR.
        if (result < 0)
            throw failure("syscall " + number);
        return result;
    }

    private int find(long start, long bytes) {
        for (int i = 0; i < starts.length; i++)
            if (starts[i] >= 0 && start >= starts[i] && start - starts[i] + bytes <= valid[i])
                return i;
        return -1;
    }

    boolean isClosed() {
        return closed;
    }

    boolean contains(long start, long bytes) {
        return !closed && find(start, bytes) >= 0;
    }

    void prepare(long[] offsets, long[] lengths, int count) throws IOException {
        prepare(offsets, lengths, count, true);
    }

    void prepare(long[] offsets, long[] lengths, int count, boolean useCache) throws IOException {
        if (closed)
            throw new IOException("Linux AIO context closed");
        Arrays.fill(pinned, false);
        // Protect hits before assigning buffers for misses in this batch.
        for (int n = 0; n < count; n++) {
            if (lengths[n] == 0)
                continue;
            int slot = find(offsets[n], lengths[n]);
            if (slot >= 0)
                pinned[slot] = true;
        }
        int pending = 0;
        for (int n = 0; n < count; n++) {
            long offset = offsets[n], length = lengths[n];
            if (length == 0 || find(offset, length) >= 0
                    || (useCache && cache != null && cache.contains(offset, length)))
                continue;
            long start = offset & -4096L;
            long span = offset - start + length;
            if (span > MAX_RECORD_BYTES)
                continue;
            int bytes = (int) ((span + 4095) & -4096L);
            boolean duplicate = false;
            for (int j = 0; j < pending; j++) {
                int slot = pendingSlots[j];
                if (start == starts[slot] && bytes == pendingBytes[j]) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
                continue;
            int slot = 0;
            while (slot < pinned.length && pinned[slot]) slot++;
            if (slot == pinned.length)
                break;
            pinned[slot] = true;
            starts[slot] = start;
            valid[slot] = 0;
            pendingSlots[pending] = slot;
            pendingBytes[pending] = bytes;
            // Linux aio_abi.h: 64-byte iocb, PREAD opcode=0; zero flags/reserved fields.
            MemorySegment cb = control.asSlice(pending * 64L, 64);
            cb.fill((byte) 0);
            cb.set(ValueLayout.JAVA_LONG, 0, pending);
            cb.set(ValueLayout.JAVA_INT, 20, fd);
            cb.set(ValueLayout.JAVA_LONG, 24, buffers[slot].address());
            cb.set(ValueLayout.JAVA_LONG, 32, bytes);
            cb.set(ValueLayout.JAVA_LONG, 40, start);
            pointers.set(ValueLayout.ADDRESS, pending * 8L, cb);
            pending++;
        }
        if (pending == 0)
            return;
        try {
            execute(pending);
        } catch (IOException e) {
            // io_destroy waits for in-flight I/O before any native memory is freed.
            try {
                close();
            } catch (IOException other) {
                e.addSuppressed(other);
            }
            throw e;
        }
    }

    private void execute(int count) throws IOException {
        int submitted = 0, completed = 0;
        IOException failure = null;
        while (completed < count) {
            if (submitted < count) {
                long n = raw(submit, context, count - submitted, pointers.address() + submitted * 8L, 0, 0);
                if (n < 0) {
                    int code = errno.get(ValueLayout.JAVA_INT, ERRNO_OFFSET);
                    if (code == 4)
                        continue;
                    if (code != 11 || submitted == completed)
                        throw failure("io_submit");
                } else {
                    if (n == 0 || n > count - submitted)
                        throw new IOException("Invalid io_submit count " + n);
                    submitted += (int) n;
                    readRequests += n;
                    submissions++;
                }
            }
            if (submitted == completed)
                continue;
            int received = (int) call(getevents, context, 1, submitted - completed, events.address(), 0);
            for (int i = 0; i < received; i++) {
                long id = events.get(ValueLayout.JAVA_LONG, i * 32L);
                long result = events.get(ValueLayout.JAVA_LONG, i * 32L + 16);
                long secondary = events.get(ValueLayout.JAVA_LONG, i * 32L + 24);
                if (id < 0 || id >= submitted)
                    throw new IOException("Invalid AIO completion ID " + id);
                int slot = pendingSlots[(int) id];
                if (valid[slot] != 0)
                    throw new IOException("Duplicate AIO completion ID " + id);
                long expected = Math.min(pendingBytes[(int) id], fileSize - starts[slot]);
                if (result != expected || secondary != 0) {
                    if (failure == null)
                        failure = new IOException("AIO read at " + starts[slot] + " returned " + result + "/"
                                + secondary + "; expected " + expected);
                } else
                    valid[slot] = (int) result;
            }
            completed += received;
        }
        if (failure != null)
            throw failure;
    }

    boolean copy(long start, int bytes, MemorySegment target) {
        if (closed)
            return false;
        int slot = find(start, bytes);
        if (slot < 0)
            return false;
        MemorySegment.copy(buffers[slot], start - starts[slot], target, 0, bytes);
        return true;
    }

    @Override
    public void close() throws IOException {
        if (closed)
            return;
        IOException error = null;
        if (context != 0) {
            try {
                call(destroy, context, 0, 0, 0, 0);
                context = 0;
            } catch (IOException e) {
                throw e;
            } // Keep buffers alive if the kernel still owns them.
        }
        closed = true;
        if (ownedFile != null) {
            try {
                ownedFile.close();
            } catch (IOException e) {
                error = e;
            }
        }
        arena.close();
        if (error != null)
            throw error;
    }
}
