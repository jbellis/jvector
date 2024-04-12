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

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;

public class BufferedRandomAccessWriter implements Closeable {
    private final RandomAccessFile raf;

    // buffer structures
    private final TransparentByteArrayOutputStream scratchBytes = new TransparentByteArrayOutputStream(4096);
    private final DataOutputStream scratch = new DataOutputStream(scratchBytes);

    public BufferedRandomAccessWriter(Path path) throws FileNotFoundException {
        raf = new RandomAccessFile(path.toFile(), "rw");
    }

    public void seek(long position) throws IOException {
        raf.seek(position);
    }

    public void writeBuffered(ChunkWriter writer) throws IOException {
        assert scratchBytes.size() == 0;
        writer.write(scratch);
        raf.write(scratchBytes.getArray(), 0, scratchBytes.size());
        scratchBytes.reset();
    }

    @Override
    public void close() throws IOException {
        raf.close();
    }

    public void writeInt(int i) throws IOException {
        raf.writeInt(i);
    }

    public long getFilePointer() throws IOException {
        return raf.getFilePointer();
    }

    // because there is still no ThrowingRunnable in the JDK
    @FunctionalInterface
    public interface ChunkWriter {
        void write(DataOutput out) throws IOException;
    }

    // silly that we need to do this but at least the JDK made `buf` available to subclasses
    private static class TransparentByteArrayOutputStream extends ByteArrayOutputStream {
        public TransparentByteArrayOutputStream(int size) {
            super(size);
        }

        public byte[] getArray() {
            return buf;
        }
    }
}
