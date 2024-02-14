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
import java.nio.ByteBuffer;

/**
 * This is a subset of DataInput, plus seek and readFully methods, which allows implementations
 * to use more efficient options like FloatBuffer for bulk reads.
 */
public interface RandomAccessReader extends AutoCloseable {
    void seek(long offset) throws IOException;

    int readInt() throws IOException;

    void readFully(byte[] bytes) throws IOException;

    void readFully(ByteBuffer buffer) throws IOException;

    void readFully(float[] floats) throws IOException;

    void readFully(long[] vector) throws IOException;

    void read(int[] ints, int offset, int count) throws IOException;

    void close() throws IOException;
}
