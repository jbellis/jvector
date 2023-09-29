package io.github.jbellis.jvector.example.util;

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

public class MMapReaderSupplier implements ReaderSupplier {
    private final MMapBuffer buffer;

    public MMapReaderSupplier(Path path) throws IOException {
        buffer = new MMapBuffer(path, FileChannel.MapMode.READ_ONLY, ByteOrder.BIG_ENDIAN);
    }

    @Override
    public RandomAccessReader get() {
        return new MMapReader(buffer);
    }

    @Override
    public void close() throws IOException {
        buffer.close();
    }
}
