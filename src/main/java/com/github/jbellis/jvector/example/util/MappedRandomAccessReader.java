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

package com.github.jbellis.jvector.example.util;

import com.github.jbellis.jvector.disk.RandomAccessReader;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * DO NOT use this for anything you care about.
 * Lies about implementing interfaces. Bare minimum I/O to run Bench/SiftSmall
 * against disk in reasonable time. Does not handle files above 2 GB.
 */
public class MappedRandomAccessReader implements RandomAccessReader {
    private final MappedByteBuffer mbb;

    public MappedRandomAccessReader(String name) throws IOException {
        var raf = new RandomAccessFile(name, "r");
        if (raf.length() > Integer.MAX_VALUE) {
            throw new RuntimeException("MappedRandomAccessReader doesn't support large files");
        }
        mbb = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length());
        mbb.load();
        raf.close();
    }

    private MappedRandomAccessReader(MappedByteBuffer sourceMbb) {
        mbb = sourceMbb;
    }

    @Override
    public void seek(long offset) {
        mbb.position((int) (offset >= mbb.limit() ? mbb.limit() : offset));
    }

    @Override
    public void readFloatsAt(long offset, float[] buffer) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = mbb.getFloat((int) offset + i);
        }
    }

    @Override
    public void readFully(byte[] b) {
        mbb.get(b);
    }

    @Override
    public void readFully(byte[] b, int off, int len) {
        mbb.get(b, off, len);
    }

    @Override
    public int skipBytes(int n) {
        var position = mbb.position();
        var limit = mbb.limit();
        var skip = position + n > limit ? limit - position : n;
        mbb.position(mbb.position() + skip);
        return skip;
    }

    @Override
    public boolean readBoolean() {
        return mbb.get() != 0;
    }

    @Override
    public byte readByte() {
        return mbb.get();
    }

    @Override
    public int readUnsignedByte() {
        return mbb.get() & 0xFF;
    }

    @Override
    public short readShort() {
        return mbb.getShort();
    }

    @Override
    public int readUnsignedShort() {
        return mbb.getShort() & 0xFFFF;
    }

    @Override
    public char readChar() {
        return mbb.getChar();
    }

    @Override
    public int readInt() {
        return mbb.getInt();
    }

    @Override
    public long readLong() {
        return mbb.getLong();
    }

    @Override
    public float readFloat() {
        return mbb.getFloat();
    }

    @Override
    public double readDouble() {
        return mbb.getDouble();
    }

    @Override
    public String readLine() {
        throw new UnsupportedOperationException("This hack job doesn't support this");
    }

    @Override
    public String readUTF() {
        throw new UnsupportedOperationException("This hack job doesn't support this");
    }

    @Override
    public void close() {

    }

    public MappedRandomAccessReader duplicate() {
        return new MappedRandomAccessReader(mbb.duplicate());
    }
}
