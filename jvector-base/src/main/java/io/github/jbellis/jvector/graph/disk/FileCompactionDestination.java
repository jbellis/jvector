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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * {@link CompactionDestination} that writes a standalone graph file at offset 0.
 * Backs {@link CompactionDestination#file(Path)}.
 */
final class FileCompactionDestination implements CompactionDestination {

    private final Path path;
    private final long startOffset;

    FileCompactionDestination(Path path) {
        this(path, 0L);
    }

    FileCompactionDestination(Path path, long startOffset) {
        if (path == null) throw new NullPointerException("path");
        this.path = path;
        this.startOffset = startOffset;
    }

    @Override
    public OutputReservation open() {
        return new OutputReservation() {
            private boolean committed;

            @Override
            public Path file() {
                return path;
            }

            @Override
            public long startOffset() {
                return startOffset;
            }

            @Override
            public void commit(long bodyLength) {
                committed = true;
            }

            @Override
            public void close() throws IOException {
                if (!committed) {
                    Files.deleteIfExists(path);
                }
            }
        };
    }
}
