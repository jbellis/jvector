package io.github.jbellis.jvector.disk;

import java.io.IOException;
import java.nio.file.Path;

public class SimpleMappedReaderSupplier implements ReaderSupplier {
    private final SimpleMappedReader smr;

    public SimpleMappedReaderSupplier(Path path) throws IOException {
        smr = new SimpleMappedReader(path);
    }

    @Override
    public RandomAccessReader get() {
        return smr.duplicate();
    }

    @Override
    public void close() {
        smr.close();
    }
};
