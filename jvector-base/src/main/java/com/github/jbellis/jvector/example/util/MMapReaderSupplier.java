package com.github.jbellis.jvector.example.util;

import com.github.jbellis.jvector.disk.RandomAccessReader;
import com.github.jbellis.jvector.disk.ReaderSupplier;
import com.indeed.util.mmap.MMapBuffer;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

public class MMapReaderSupplier implements ReaderSupplier {
    private final MMapBuffer buffer;

    public MMapReaderSupplier(Path path) throws IOException {
        buffer = new MMapBuffer(path, FileChannel.MapMode.READ_ONLY, ByteOrder.LITTLE_ENDIAN);
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
