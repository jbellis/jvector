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

/**
 * This is a subset of DataInput, plus seek and readFully(float[]), which allows implementations
 * to use a more efficient option like FloatBuffer.
 */
public interface RandomAccessReader extends AutoCloseable {
    public void seek(long offset) throws IOException;

    public int readInt() throws IOException;

    public void readFully(byte[] bytes) throws IOException;

    public void readFully(float[] floats) throws IOException;

    void close() throws IOException;
}
