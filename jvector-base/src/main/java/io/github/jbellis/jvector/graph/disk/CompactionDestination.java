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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.annotations.Experimental;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Embedding extension point: tells {@link OnDiskGraphIndexCompactor} where to write its
 * compacted graph so the body lands directly inside the caller's container file (after a
 * header the caller reserves) — eliminating the need for a temp file and a second copy pass.
 *
 * <p>The compactor requires a real file: it writes node records at computed absolute byte
 * offsets via random-access I/O, and memory-maps the output for the pre-encode PQ code
 * cache during refinement. A sequential stream is not sufficient.
 *
 * <p>Usage:
 * <pre>{@code
 * CompactionDestination destination = () -> new CompactionDestination.OutputReservation() {
 *     public Path file()           { return myContainerPath; }
 *     public long startOffset()    { return myHeaderSize; }
 *     public void commit(long len) { recordBodyLength(len); }
 *     public void close()          { }
 * };
 * compactor.compact(destination);
 * }</pre>
 *
 * @see CompactionDestination#file(Path) for the simple standalone-file case
 */
@FunctionalInterface
@Experimental
public interface CompactionDestination {

    /**
     * Opens a fresh output reservation for one compaction call.
     * Called exactly once by the compactor at the start of {@code compact(CompactionDestination)}.
     */
    OutputReservation open() throws IOException;

    /**
     * One compaction's output region and its commit/abort lifecycle.
     * The compactor opens one instance, writes the graph body into it,
     * calls {@link #commit} exactly once on success, then always calls {@link #close}.
     */
    interface OutputReservation extends AutoCloseable {

        /**
         * The container file the graph body is written into.
         * Must be a real file path; the compactor uses random-access and mmap on it.
         */
        Path file();

        /**
         * The byte offset within {@link #file()} at which the graph body begins.
         * The region {@code [0, startOffset)} is left untouched by the compactor,
         * allowing the caller to place its own header there.
         */
        long startOffset();

        /**
         * Called exactly once, on the success path, after the body has been fully written.
         * Reports the byte length of the graph body ({@code file() size - startOffset()},
         * measured after any transient sections such as the PQ code cache have been removed).
         * The caller may finalize its container here (e.g. write a footer or checksum).
         * Must be called before {@link #close()}.
         */
        void commit(long bodyLength) throws IOException;

        /**
         * Always called (try-with-resources), after {@link #commit} on success or directly on
         * failure. A close without a prior commit means the compaction failed; the caller should
         * discard the partial output. Releases any caller-owned resources.
         */
        @Override
        void close() throws IOException;
    }

    /**
     * Creates a standalone destination that writes a self-contained graph file at offset 0.
     * {@code commit} is a no-op; if {@code close} is reached without a prior {@code commit}
     * the partial file is deleted.
     */
    static CompactionDestination file(Path path) {
        return new FileCompactionDestination(path);
    }

    /**
     * Creates a standalone destination that writes into an existing file starting at
     * {@code startOffset}, leaving the prefix region {@code [0, startOffset)} untouched.
     * {@code commit} is a no-op; if {@code close} is reached without a prior {@code commit}
     * the partial file is deleted.
     */
    static CompactionDestination file(Path path, long startOffset) {
        return new FileCompactionDestination(path, startOffset);
    }
}
