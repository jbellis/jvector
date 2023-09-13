package com.github.jbellis.jvector.example.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.github.jbellis.jvector.disk.RandomAccessReader;
import com.github.jbellis.jvector.disk.ReaderSupplier;

public class ReaderSupplierFactory {
    public static ReaderSupplier open(Path path) throws IOException {
        try {
            return new MMapReaderSupplier(path);
        } catch (UnsatisfiedLinkError e) {
            if (Files.size(path) > Integer.MAX_VALUE) {
                throw new RuntimeException("File sizes greater than 2GB are not supported on Windows--contributions welcome");
            }

            return new ReaderSupplier() {
                private final SimpleMappedReader smr = new SimpleMappedReader(path);

                @Override
                public RandomAccessReader get() {
                    return smr.duplicate();
                }

                @Override
                public void close() {
                    smr.close();
                }
            };
        }
    }
}
