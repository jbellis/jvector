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
import java.nio.channels.FileChannel;

/**
 * {@link ReaderSupplier} that vends {@link FileChannelReader} instances backed by a shared
 * {@link FileChannel}.
 *
 * <p>Each call to {@link #get()} returns a new reader positioned at offset 0. All vended
 * readers share the same underlying channel and may be used concurrently: positional
 * {@link FileChannel#read(java.nio.ByteBuffer, long)} is thread-safe by spec.
 *
 * <p>The channel is <em>not</em> closed by {@link #close()}; ownership stays with whoever
 * opened it. Callers must not close the channel while readers vended by this supplier are
 * still in use.
 */
public class FileChannelReaderSupplier implements ReaderSupplier {

    private final FileChannel channel;

    /**
     * @param channel open, readable {@link FileChannel}; not owned by this supplier
     */
    public FileChannelReaderSupplier(FileChannel channel) {
        this.channel = channel;
    }

    @Override
    public RandomAccessReader get() {
        return new FileChannelReader(channel, 0);
    }

    /** No-op: channel ownership belongs to the caller. */
    @Override
    public void close() throws IOException {
        // channel lifecycle is managed externally
    }
}
