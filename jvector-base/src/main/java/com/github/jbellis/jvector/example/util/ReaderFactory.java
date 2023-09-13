package com.github.jbellis.jvector.example.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.github.jbellis.jvector.disk.RandomAccessReader;

public class ReaderFactory {
    public static RandomAccessReader open(Path path) throws IOException {
        try {
            return new MMapBufferReader(path);
        } catch (UnsatisfiedLinkError e) {
            if (Files.size(path) > Integer.MAX_VALUE) {
                throw new RuntimeException("File sizes greater than 2GB are not supported on Windows--contributions welcome");
            }
            return new SimpleMappedReader(path.toString());
        }
    }

    public static RandomAccessReader duplicate(RandomAccessReader reader) throws IOException {
        if (reader instanceof MMapBufferReader) {
            return new MMapBufferReader(((MMapBufferReader) reader).getPath());
        } else if (reader instanceof SimpleMappedReader) {
            return ((SimpleMappedReader) reader).duplicate();
        }
        throw new IllegalStateException();
    }
}
