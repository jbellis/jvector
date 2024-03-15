package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.vector.ArrayVectorFloat;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.file.Path;

// no buffering, slow af
public class SimpleWriter implements RandomAccessWriter {
    private final RandomAccessFile raf;

    public SimpleWriter(Path path) throws FileNotFoundException {
        raf = new RandomAccessFile(path.toFile(), "rw");
    }

    @Override
    public void seek(long position) throws IOException {
        raf.seek(position);
    }

    @Override
    public void writeFully(VectorFloat<?> vector) throws IOException {
        // TODO I can't find a way to do this without copying the data
        // (I think this is still modestly faster than writing float-at-a-time)
        float[] data = ((ArrayVectorFloat) vector).get();
        var bb = ByteBuffer.allocate(data.length * 4);
        bb.asFloatBuffer().put(data);
        raf.getChannel().write(bb);
    }

    @Override
    public void close() throws Exception {
        raf.close();
    }
}
