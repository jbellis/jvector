package com.github.jbellis.jvector.example.util;

import com.github.jbellis.jvector.disk.ReaderSupplier;
import com.github.jbellis.jvector.disk.SimpleMappedReaderSupplier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ReaderSupplierFactory {
    public static ReaderSupplier open(Path path) throws IOException {
        try {
            return new MMapReaderSupplier(path);
        } catch (UnsatisfiedLinkError e) {
            if (Files.size(path) > Integer.MAX_VALUE) {
                throw new RuntimeException("File sizes greater than 2GB are not supported on Windows--contributions welcome");
            }

            return new SimpleMappedReaderSupplier(path);
        }
    }
}
