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

import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Optional Linux O_DIRECT reader for immutable files. Each view owns one aligned
 * buffer bounded to 64KiB. Positional reads share a descriptor but never a cursor or buffer.
 * The supplier can enable bounded kernel AIO and a shared sector cache explicitly.
 */
public final class LinuxDirectReader implements RandomAccessReader {
    private static final int BLOCK_BYTES = 4096;
    private static final int WINDOW_BYTES = 64 * 1024;
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private final Supplier supplier;
    private final Arena arena;
    private final MemorySegment memory;
    private final ByteBuffer buffer;
    private final MemorySegment errno;
    private final DirectBlockCache cache;
    private final LinuxAioBatch aio;
    private final long[] singleOffset = new long[1], singleLength = new long[1];
    private long position;
    private long blockStart = -1;
    private int validBytes;
    private volatile boolean closed;

    private LinuxDirectReader(Supplier supplier, boolean useCache) throws IOException {
        this.supplier = supplier;
        this.cache = useCache ? supplier.cache : null;
        arena = Arena.ofShared();
        try {
            memory = arena.allocate(WINDOW_BYTES, BLOCK_BYTES);
            buffer = memory.asByteBuffer();
            errno = arena.allocate(LinuxDirectFile.ERRNO);
            aio = supplier.ioDepth == 0 ? null : new LinuxAioBatch(supplier.file, supplier.ioDepth, cache);
        } catch (Throwable t) {
            arena.close();
            throw t;
        }
    }

    private void ensureOpen() throws IOException {
        if (closed || !supplier.file.isOpen() || (aio != null && aio.isClosed()))
            throw new IOException("Direct reader is closed");
    }

    private void checkRead(long bytes) throws IOException {
        ensureOpen();
        if (bytes > supplier.fileSize - position)
            throw new EOFException("Read past end at " + position);
    }

    private void loadRange(long start, int bytes) throws IOException {
        // Invalidate before overwriting: an I/O error must not expose stale metadata.
        blockStart = -1;
        validBytes = 0;
        int expected = (int) Math.min(bytes, supplier.fileSize - start);
        int n;
        if (cache != null && cache.copy(start, expected, memory)) {
            n = expected;
        } else if (aio == null) {
            n = supplier.file.read(errno, memory, bytes, start);
        } else {
            if (!aio.copy(start, expected, memory)) {
                singleOffset[0] = start;
                singleLength[0] = expected;
                aio.prepare(singleOffset, singleLength, 1, false);
                if (!aio.copy(start, expected, memory))
                    throw new IOException("AIO range not completed");
            }
            n = expected;
        }
        if (n != expected)
            throw new EOFException("Immutable direct file changed or short read at " + start);
        if (cache != null)
            cache.put(start, memory, n);
        validBytes = n;
        blockStart = start;
    }

    @Override
    public void prepareRead(long offset, long length) throws IOException {
        ensureOpen();
        if (offset < 0 || length < 0 || offset > supplier.fileSize || length > supplier.fileSize - offset)
            throw new EOFException("Invalid prepared range at " + offset);
        if (length == 0)
            return;
        if (blockStart >= 0 && offset >= blockStart && offset + length <= blockStart + validBytes)
            return;
        long start = offset & -((long) BLOCK_BYTES);
        long span = offset - start + length;
        // Oversized records use the ordinary block path; view memory stays bounded.
        if (span > WINDOW_BYTES)
            return;
        int bytes = (int) ((span + BLOCK_BYTES - 1) & -((long) BLOCK_BYTES));
        loadRange(start, bytes);
    }

    @Override
    public int readAheadLimit() {
        return supplier.ioDepth;
    }

    @Override
    public boolean isPrepared(long offset, long length) {
        return !closed
                && ((blockStart >= 0 && offset >= blockStart && offset - blockStart + length <= validBytes)
                        || (aio != null && aio.contains(offset, length))
                        || (cache != null && cache.contains(offset, length)));
    }

    @Override
    public void prepareReads(long[] offsets, long[] lengths, int count) throws IOException {
        Objects.checkFromIndexSize(0, count, offsets.length);
        Objects.checkFromIndexSize(0, count, lengths.length);
        ensureOpen();
        for (int i = 0; i < count; i++) {
            if (offsets[i] < 0 || lengths[i] < 0 || offsets[i] > supplier.fileSize
                    || lengths[i] > supplier.fileSize - offsets[i])
                throw new EOFException("Invalid prepared range at " + offsets[i]);
        }
        if (aio == null)
            RandomAccessReader.super.prepareReads(offsets, lengths, count);
        else
            aio.prepare(offsets, lengths, count);
    }

    private int blockOffset() throws IOException {
        if (blockStart < 0 || position < blockStart || position - blockStart >= validBytes) {
            loadRange(position & -((long) BLOCK_BYTES), BLOCK_BYTES);
        }
        return (int) (position - blockStart);
    }

    private long readBits(int bytes) throws IOException {
        long result = 0;
        for (int i = 0; i < bytes; i++) {
            result = (result << 8) | (memory.get(ValueLayout.JAVA_BYTE, blockOffset()) & 0xffL);
            position++;
        }
        return result;
    }

    @Override
    public void seek(long offset) throws IOException {
        ensureOpen();
        if (offset < 0 || offset > supplier.fileSize)
            throw new EOFException("Invalid seek: " + offset);
        position = offset;
    }

    @Override
    public long getPosition() throws IOException {
        ensureOpen();
        return position;
    }

    @Override
    public long length() throws IOException {
        ensureOpen();
        return supplier.fileSize;
    }

    @Override
    public int readInt() throws IOException {
        checkRead(4);
        int off = blockOffset();
        if (off + 4 > validBytes)
            return (int) readBits(4);
        int value = memory.get(INT, off);
        position += 4;
        return value;
    }

    @Override
    public long readLong() throws IOException {
        checkRead(8);
        int off = blockOffset();
        if (off + 8 > validBytes)
            return readBits(8);
        long value = memory.get(LONG, off);
        position += 8;
        return value;
    }

    @Override
    public float readFloat() throws IOException {
        return Float.intBitsToFloat(readInt());
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        checkRead(bytes.length);
        int done = 0;
        while (done < bytes.length) {
            int off = blockOffset();
            int count = Math.min(bytes.length - done, validBytes - off);
            buffer.get(off, bytes, done, count);
            position += count;
            done += count;
        }
    }

    @Override
    public void readFully(ByteBuffer target) throws IOException {
        if (target.isReadOnly())
            throw new java.nio.ReadOnlyBufferException();
        checkRead(target.remaining());
        while (target.hasRemaining()) {
            int off = blockOffset();
            int count = Math.min(target.remaining(), validBytes - off);
            target.put(target.position(), buffer, off, count);
            target.position(target.position() + count);
            position += count;
        }
    }

    @Override
    public void read(int[] values, int offset, int count) throws IOException {
        Objects.checkFromIndexSize(offset, count, values.length);
        checkRead(count * 4L);
        while (count > 0) {
            int off = blockOffset();
            int n = Math.min(count, (validBytes - off) / 4);
            if (n == 0) {
                values[offset++] = readInt();
                count--;
                continue;
            }
            MemorySegment.copy(memory, INT, off, values, offset, n);
            position += n * 4L;
            offset += n;
            count -= n;
        }
    }

    @Override
    public void read(float[] values, int offset, int count) throws IOException {
        Objects.checkFromIndexSize(offset, count, values.length);
        checkRead(count * 4L);
        while (count > 0) {
            int off = blockOffset();
            int n = Math.min(count, (validBytes - off) / 4);
            if (n == 0) {
                values[offset++] = readFloat();
                count--;
                continue;
            }
            MemorySegment.copy(memory, FLOAT, off, values, offset, n);
            position += n * 4L;
            offset += n;
            count -= n;
        }
    }

    @Override
    public void readFully(long[] values) throws IOException {
        checkRead(values.length * 8L);
        int offset = 0;
        while (offset < values.length) {
            int off = blockOffset();
            int n = Math.min(values.length - offset, (validBytes - off) / 8);
            if (n == 0) {
                values[offset++] = readLong();
                continue;
            }
            MemorySegment.copy(memory, LONG, off, values, offset, n);
            position += n * 8L;
            offset += n;
        }
    }

    @Override
    public void close() throws IOException {
        synchronized (supplier) {
            if (!closed) {
                if (aio != null) {
                    aio.close();
                    supplier.retiredRequests += aio.readRequests;
                    supplier.retiredSubmissions += aio.submissions;
                }
                closed = true;
                arena.close();
                supplier.readers.remove(this);
            }
        }
    }

    /** Owns the descriptor and closes all outstanding views. Close after readers are idle. */
    public static final class Supplier implements ReaderSupplier {
        private final LinuxDirectFile file;
        private final long fileSize;
        private final int ioDepth;
        private final DirectBlockCache cache;
        private long retiredRequests, retiredSubmissions;
        private final Set<LinuxDirectReader> readers = new HashSet<>();

        public Supplier(Path path) throws IOException {
            this(path, 0);
        }

        /** ioDepth=0 is synchronous; 1..32 uses bounded kernel AIO per reader. */
        public Supplier(Path path, int ioDepth) throws IOException {
            this(path, ioDepth, 0);
        }

        public Supplier(Path path, int ioDepth, long cacheBytes) throws IOException {
            if (ioDepth < 0 || ioDepth > 32)
                throw new IllegalArgumentException("I/O depth must be 0..32");
            this.ioDepth = ioDepth;
            if (!System.getProperty("os.name").equals("Linux"))
                throw new IOException("Direct reader requires Linux");
            file = new LinuxDirectFile(path.toAbsolutePath());
            try {
                fileSize = file.size;
                cache = cacheBytes == 0 ? null : new DirectBlockCache(cacheBytes);
            } catch (Throwable t) {
                file.close();
                throw t;
            }
        }

        @Override
        public synchronized LinuxDirectReader get() throws IOException {
            return getReader(true);
        }

        @Override
        public synchronized LinuxDirectReader getUncached() throws IOException {
            return getReader(false);
        }

        private LinuxDirectReader getReader(boolean useCache) throws IOException {
            if (!file.isOpen())
                throw new IOException("Direct supplier is closed");
            var reader = new LinuxDirectReader(this, useCache);
            readers.add(reader);
            return reader;
        }

        @Override
        public synchronized void close() throws IOException {
            IOException error = null;
            for (var reader : List.copyOf(readers)) {
                try {
                    reader.close();
                } catch (IOException e) {
                    if (error == null)
                        error = e;
                    else
                        error.addSuppressed(e);
                }
            }
            if (error != null)
                throw error;
            try {
                file.close();
            } finally {
                if (cache != null) cache.close();
            }
        }

        /** Read after workers are idle for an exact snapshot; live counters are diagnostic estimates. */
        public synchronized Statistics statistics() {
            long requests = retiredRequests + file.synchronousRequests.sum();
            long submissions = retiredSubmissions + file.synchronousRequests.sum();
            for (var reader : readers)
                if (reader.aio != null) {
                    requests += reader.aio.readRequests;
                    submissions += reader.aio.submissions;
                }
            return new Statistics(requests, submissions, cache == null ? 0 : cache.hits.sum(),
                    cache == null ? 0 : cache.misses.sum(), cache == null ? 0 : cache.evictions.sum(),
                    cache == null ? 0 : cache.reservedBytes,
                    readers.size() * (WINDOW_BYTES + ioDepth * (long) LinuxAioBatch.MAX_RECORD_BYTES));
        }
    }

    /** Cache bytes include a metadata allowance; view bytes report native payload, not kernel context memory. */
    public record Statistics(long ioRequests, long ioSubmissions, long cacheHits, long cacheMisses, long cacheEvictions,
            long cacheReservedBytes, long viewBufferBytes) {}
}
