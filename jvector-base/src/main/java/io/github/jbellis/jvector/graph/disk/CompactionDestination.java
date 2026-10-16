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
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Tells {@link OnDiskGraphIndexCompactor} where to write its compacted graph so the body
 * lands directly inside the caller's container file (after a header the caller reserves) —
 * eliminating the need for a temp file and a second copy pass.
 *
 * <p>The compactor works entirely through {@link #channel()}: sequential writes,
 * positional random-access writes, mmap, truncate, and size measurement all go through
 * the same open channel with no additional OS opens.
 *
 * <p>Usage:
 * <pre>{@code
 * // write header through the channel, leaving it positioned right after
 * long startOffset = myOpenChannel.position();
 * var destination = CompactionDestination.channel(myOpenChannel);
 * compactor.compact(destination);
 * long bodyLength = myOpenChannel.size() - startOffset;
 * }</pre>
 *
 * @see #file(Path)       standalone file
 * @see #channel(FileChannel)  embed in a caller-managed channel
 */
@Experimental
public class CompactionDestination {

    @FunctionalInterface
    private interface IORunnable {
        void run() throws IOException;
    }

    private final FileChannel channel;
    private final IORunnable onCommit;

    private CompactionDestination(FileChannel channel, IORunnable onCommit) {
        this.channel = channel;
        this.onCommit = onCommit;
    }

    /** The open {@link FileChannel} the graph body is written into. */
    public FileChannel channel() {
        return channel;
    }

    /** Called by the compactor when it is done writing. */
    void commit() throws IOException {
        onCommit.run();
    }

    /**
     * Opens a new {@link FileChannel} on {@code path} at offset 0.
     */
    public static CompactionDestination file(Path path) {
        try {
            var fc = FileChannel.open(path,
                    StandardOpenOption.READ,
                    StandardOpenOption.WRITE,
                    StandardOpenOption.CREATE);
            return new CompactionDestination(fc, fc::close);
        } catch (IOException e) {
            throw new RuntimeException("Failed to open compaction destination: " + path, e);
        }
    }

    /**
     * Wraps a caller-supplied {@link FileChannel}.
     * The caller retains ownership of the channel.
     */
    public static CompactionDestination channel(FileChannel channel) {
        return new CompactionDestination(channel, () -> {});
    }
}
