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
package io.github.jbellis.jvector.example.util;

import com.indeed.util.mmap.MMapBuffer;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;

public class MMapReaderSupplier implements ReaderSupplier {
    private final MMapBuffer buffer;

    public MMapReaderSupplier(Path path) throws IOException {
        buffer = new MMapBuffer(path, FileChannel.MapMode.READ_ONLY, ByteOrder.BIG_ENDIAN);
    }

    @Override
    public RandomAccessReader get() {
        return new MMapReader(buffer);
    }

    @Override
    public void close() throws IOException {
        buffer.close();
    }
}
