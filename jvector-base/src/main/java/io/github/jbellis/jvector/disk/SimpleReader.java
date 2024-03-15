package io.github.jbellis.jvector.disk;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

// TODO what are the low-hanging optimization options here?
// The requirement that we need to read from a file that is potentially changing in length
// limits our options.
public class SimpleReader implements RandomAccessReader {
    RandomAccessFile raf;

    public SimpleReader(Path path) throws FileNotFoundException {
        raf = new RandomAccessFile(path.toFile(), "r");
    }

    @Override
    public void seek(long offset) throws IOException {
        raf.seek(offset);
    }

    @Override
    public int readInt() throws IOException {
        return raf.readInt();
    }

    @Override
    public void readFully(byte[] bytes) throws IOException {
        raf.readFully(bytes);
    }

    @Override
    public void readFully(ByteBuffer buffer) throws IOException {
        raf.getChannel().read(buffer);
    }

    @Override
    public void readFully(float[] floats) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(floats.length * Float.BYTES);
        raf.getChannel().read(bb);
        bb.flip().order(ByteOrder.BIG_ENDIAN);
        bb.asFloatBuffer().get(floats);
    }

    @Override
    public void readFully(long[] vector) throws IOException {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = raf.readLong();
        }
    }

    @Override
    public void read(int[] ints, int offset, int count) throws IOException {
        for (int i = 0; i < count; i++) {
            ints[i] = raf.readInt();
        }
    }

    @Override
    public void close() throws IOException {
        raf.close();
    }
}
