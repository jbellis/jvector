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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

public class MemorySegmentReaderSupplier implements ReaderSupplier {
    private final InternalMemorySegmentReader reader;

    public MemorySegmentReaderSupplier(Path path) throws IOException {
        reader = new InternalMemorySegmentReader(path);
    }

    @Override
    public RandomAccessReader get() {
        return reader.duplicate();
    }

    @Override
    public void close() {
        reader.close();
    }

    private static class InternalMemorySegmentReader extends MemorySegmentReader {

        private final boolean shouldClose;

        private InternalMemorySegmentReader(Path path) throws IOException {
            super(path);
            shouldClose = true;
        }

        private InternalMemorySegmentReader(Arena arena, MemorySegment memory) {
            super(arena, memory);
            shouldClose = false;
        }

        @Override
        public void close() {
            if (shouldClose) {
                super.close();
            }
        }

        @Override
        public InternalMemorySegmentReader duplicate() {
            return new InternalMemorySegmentReader(arena, memory);
        }
    }
}
