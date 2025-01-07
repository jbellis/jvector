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
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfFloat;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link MemorySegment} based implementation of RandomAccessReader.  This is the recommended
 * RandomAccessReader implementation included with JVector.
 * <p>
 * MemorySegmentReader applies MADV_RANDOM to the backing storage, and doesn't have the 2GB file size limitation
 * of {@link SimpleMappedReader}.
 */
public class MemorySegmentReader implements RandomAccessReader {
    private static final Logger logger = LoggerFactory.getLogger(MemorySegmentReader.class);

    private static final int MADV_RANDOM = 1; // Value for Linux
    private static final OfInt intLayout = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfFloat floatLayout = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfLong longLayout = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);

    final Arena arena;
    final MemorySegment memory;
    private long position = 0;

    public MemorySegmentReader(Path path) throws IOException {
        arena = Arena.ofShared();
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            memory = ch.map(MapMode.READ_ONLY, 0L, ch.size(), arena);
            
            // Apply MADV_RANDOM advice
            var linker = Linker.nativeLinker();
            var maybeMadvise = linker.defaultLookup().find("posix_madvise");
            if (maybeMadvise.isPresent()) {
                var madvise = linker.downcallHandle(maybeMadvise.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
                try
                {
                    int result = (int) madvise.invokeExact(memory, memory.byteSize(), MADV_RANDOM);
                    if (result != 0)
                    {
                        throw new IOException("posix_madvise failed with error code: " + result);
                    }
                }
                catch (Throwable t)
                {
                    throw new RuntimeException(t);
                }
            } else {
                logger.warn("posix_madvise not found, MADV_RANDOM advice not applied");
            }
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    MemorySegmentReader(Arena arena, MemorySegment memory) {
        this.arena = arena;
        this.memory = memory;
    }

    @Override
    public void seek(long offset) {
        this.position = offset;
    }

    @Override
    public long getPosition() {
        return position;
    }

    @Override
    public void readFully(float[] buffer) {
        MemorySegment.copy(memory, floatLayout, position, buffer, 0, buffer.length);
        position += buffer.length * 4L;
    }

    @Override
    public void readFully(byte[] b) {
        MemorySegment.copy(memory, ValueLayout.JAVA_BYTE, position, b, 0, b.length);
        position += b.length;
    }

    @Override
    public void readFully(ByteBuffer buffer) {
        var remaining = buffer.remaining();
        var slice = memory.asSlice(position, remaining).asByteBuffer();
        buffer.put(slice);
        position += remaining;
    }

    @Override
    public void readFully(long[] vector) {
        MemorySegment.copy(memory, longLayout, position, vector, 0, vector.length);
        position += vector.length * 8L;
    }

    @Override
    public int readInt() {
        var k = memory.get(intLayout, position);
        position += 4;
        return k;
    }

    @Override
    public float readFloat() {
        var f = memory.get(floatLayout, position);
        position += 4;
        return f;
    }

    @Override
    public void read(int[] ints, int offset, int count) {
        MemorySegment.copy(memory, intLayout, position, ints, offset, count);
        position += count * 4L;
    }

    @Override
    public void read(float[] floats, int offset, int count) {
        MemorySegment.copy(memory, floatLayout, position, floats, offset, count);
        position += count * 4L;
    }

    /**
     * Loads the contents of the mapped segment into physical memory.
     * This is a best-effort mechanism.
     */
    public void loadMemory() {
        memory.load();
    }

    @Override
    public void close() {
        arena.close();
    }

    /**
     * Creates a shallow copy of the MemorySegmentReader.
     * Underlying memory-mapped region will be shared between all copies.
     * When MemorySegmentReader is closed, all copies will become invalid.
     */
    public MemorySegmentReader duplicate() {
        return new MemorySegmentReader(arena, memory);
    }

    public static class Supplier implements ReaderSupplier {
        private final InternalMemorySegmentReader reader;

        public Supplier(Path path) throws IOException {
            reader = new InternalMemorySegmentReader(path);
        }

        @Override
        public RandomAccessReader get() {
            return reader.duplicate();
        }

        @Override
        public void close() {
            reader.close();
        }

        private static class InternalMemorySegmentReader extends MemorySegmentReader {
            private final boolean shouldClose;

            private InternalMemorySegmentReader(Path path) throws IOException {
                super(path);
                shouldClose = true;
            }

            private InternalMemorySegmentReader(Arena arena, MemorySegment memory) {
                super(arena, memory);
                shouldClose = false;
            }

            @Override
            public void close() {
                if (shouldClose) {
                    super.close();
                }
            }

            @Override
            public InternalMemorySegmentReader duplicate() {
                return new InternalMemorySegmentReader(arena, memory);
            }
        }
    }
}
