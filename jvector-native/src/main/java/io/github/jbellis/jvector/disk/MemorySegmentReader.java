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

/**
 * {@link MemorySegment} based implementation of RandomAccessReader.
 * MemorySegmentReader doesn't have 2GB file size limitation of {@link SimpleMappedReader}.
 */
public class MemorySegmentReader implements RandomAccessReader {

    private static final OfInt intLayout = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfFloat floatLayout = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);
    private static final OfLong longLayout = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);

    private final Arena arena;
    private final MemorySegment memory;
    private long position = 0;

    public MemorySegmentReader(Path path) throws IOException {
        arena = Arena.ofShared();
        try (var ch = FileChannel.open(path, StandardOpenOption.READ)) {
            memory = ch.map(MapMode.READ_ONLY, 0L, ch.size(), arena);
        } catch (Exception e) {
            arena.close();
            throw e;
        }
    }

    private MemorySegmentReader(Arena arena, MemorySegment memory) {
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

    public MemorySegmentReader duplicate() {
        return new MemorySegmentReader(arena, memory);
    }
}
