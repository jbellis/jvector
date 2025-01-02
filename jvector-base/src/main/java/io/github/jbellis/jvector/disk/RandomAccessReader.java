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
 *
 * RandomAccessReader instances are expected to be stateful and NOT threadsafe.
 */
public interface RandomAccessReader extends AutoCloseable {
    void seek(long offset) throws IOException;

    long getPosition() throws IOException;

    int readInt() throws IOException;

    float readFloat() throws IOException;

    void readFully(byte[] bytes) throws IOException;

    void readFully(ByteBuffer buffer) throws IOException;

    default void readFully(float[] floats) throws IOException {
        read(floats, 0, floats.length);
    }

    void readFully(long[] vector) throws IOException;

    /**
     * Read `count` ints into `ints` starting at `offset` within the array.
     * Note: currently this is only called with offset=0, the extra parameter
     * is included for symmetry with the float[] overload.
     */
    void read(int[] ints, int offset, int count) throws IOException;

    /**
     * Read an int[] from each specified position in the backing file.
     * The length of each int[] will be read; currently these do not have to be the same.
     */
    default void read(int[][] ints, long[] positions) throws IOException {
        if (ints.length != positions.length) {
            throw new IllegalArgumentException(String.format("ints.length %d != positions.length %d",
                                                             ints.length, positions.length));
        }

        for (int i = 0; i < ints.length; i++) {
            seek(positions[i]);
            read(ints[i], 0, ints[i].length);
        }
    }

    /**
     * Read `count` floats into `floats`, starting at `offset` within the array.
     */
    void read(float[] floats, int offset, int count) throws IOException;

    void close() throws IOException;
}
