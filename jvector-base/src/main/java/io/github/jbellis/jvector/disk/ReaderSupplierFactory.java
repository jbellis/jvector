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

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Factory for creating ReaderSupplier instances.
 */
public class ReaderSupplierFactory {
    private static final Logger LOG = Logger.getLogger(ReaderSupplierFactory.class.getName());
    private static final String MEMORY_SEGMENT_READER_CLASSNAME = "io.github.jbellis.jvector.disk.MemorySegmentReader$Supplier";
    private static final String MMAP_READER_CLASSNAME = "io.github.jbellis.jvector.example.util.MMapReader$Supplier";

    /**
     * Private constructor to prevent instantiation.
     */
    private ReaderSupplierFactory() {
    }

    /**
     * Opens the optional Linux direct-I/O backend (JDK 22+ native module).
     * Unsupported configurations fail explicitly; this never falls back to mapping.
     */
    public static ReaderSupplier openDirect(Path path) throws IOException {
        return openDirect(path, 8, 0);
    }

    /** Open direct reads with a bounded per-view kernel AIO depth (0 selects synchronous reads). */
    public static ReaderSupplier openDirect(Path path, int ioDepth) throws IOException {
        return openDirect(path, ioDepth, 0);
    }

    /** Shared cache budget includes payload and a metadata allowance; zero disables the cache. */
    public static ReaderSupplier openDirect(Path path, int ioDepth, long cacheBytes) throws IOException {
        try {
            var type = Class.forName("io.github.jbellis.jvector.disk.LinuxDirectReader$Supplier");
            return (ReaderSupplier) type.getConstructor(Path.class, int.class, long.class).newInstance(path, ioDepth, cacheBytes);
        } catch (java.lang.reflect.InvocationTargetException e) {
            if (e.getCause() instanceof IOException) throw (IOException) e.getCause();
            throw new IOException("Unable to open direct-I/O reader", e.getCause());
        } catch (ReflectiveOperationException | LinkageError e) {
            throw new IOException("Direct-I/O reader requires the JDK 22+ native module", e);
        }
    }

    /**
     * Opens a ReaderSupplier for the given path.
     * @param path the path to open
     * @return a ReaderSupplier for the path
     * @throws IOException if an I/O error occurs
     */
    public static ReaderSupplier open(Path path) throws IOException {
        try {
            // prefer MemorySegmentReader (available under JDK 20+)
            var supplierClass = Class.forName(MEMORY_SEGMENT_READER_CLASSNAME);
            Constructor<?> ctor = supplierClass.getConstructor(Path.class);
            return (ReaderSupplier) ctor.newInstance(path);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "MemorySegmentReaderSupplier not available, falling back to MMapReaderSupplier. Reason: {0}: {1}",
                    new Object[]{e.getClass().getName(), e.getMessage()});
        }

        try {
            // fall back to MMapReader (requires a 3rd party linux-only native mmap library that is only included
            // in the build with jvector-example; this allows Bench to not embarrass us on older JDKs)
            var supplierClass = Class.forName(MMAP_READER_CLASSNAME);
            Constructor<?> ctor = supplierClass.getConstructor(Path.class);
            return (ReaderSupplier) ctor.newInstance(path);
        } catch (Exception e) {
            LOG.log(Level.WARNING, "MMapReaderSupplier not available, falling back to MappedChunkReader. More details available at level FINE.");
            LOG.log(Level.FINE, "MMapReaderSupplier instantiation exception:", e);

            // finally, fall back to MappedChunkReader
            return new MappedChunkReader.Supplier(path);
        }
    }
}
