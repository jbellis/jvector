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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * {@link RandomAccessReader} backed by a shared {@link FileChannel}.
 *
 * <p>Each instance tracks its own logical position; reads are delegated to
 * {@link FileChannel#read(ByteBuffer, long)}, which is positional (does not move the
 * channel's own position) and thread-safe by spec (JSR-51). Multiple
 * {@code FileChannelReader} instances may therefore share one open {@link FileChannel}
 * safely from concurrent threads without extra synchronisation.
 *
 * <p>The channel is <em>not</em> owned by this reader: {@link #close()} is a no-op.
 * Lifecycle of the underlying channel is the caller's responsibility.
 */
public class FileChannelReader implements RandomAccessReader {

    private final FileChannel channel;
    private long position;

    /**
     * Creates a reader positioned at {@code initialPosition}.
     *
     * @param channel         shared, open {@link FileChannel} (must support reads)
     * @param initialPosition byte offset at which reading starts
     */
    public FileChannelReader(FileChannel channel, long initialPosition) {
        this.channel = channel;
        this.position = initialPosition;
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
    public long length() throws IOException {
        return channel.size();
    }

    // ---- primitive reads ------------------------------------------------

    @Override
    public int readInt() throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        return buf.flip().getInt();
    }

    @Override
    public long readLong() throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        return buf.flip().getLong();
    }

    @Override
    public float readFloat() throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        return buf.flip().getFloat();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(bytes);
        readFully(buf);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            int n = channel.read(buffer, position);
            if (n < 0) throw new java.io.EOFException();
            position += n;
        }
    }

    @Override
    public void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    @Override
    public void readFully(long[] longs) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(longs.length * Long.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        buf.flip().asLongBuffer().get(longs);
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(count * Integer.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        buf.flip().asIntBuffer().get(ints, offset, count);
    }

    @Override
    public void read(float[] floats, int offset, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(count * Float.BYTES).order(ByteOrder.BIG_ENDIAN);
        readFully(buf);
        buf.flip().asFloatBuffer().get(floats, offset, count);
    }

    /** No-op: the channel is owned by the supplier, not this reader. */
    @Override
    public void close() {
        // channel lifecycle is managed by the owning FileChannelReaderSupplier
    }
}
