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

import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Base header for OnDiskGraphIndex functionality.
 */
class CommonHeader {
    public final int version;
    public final int size;
    public final int dimension;
    public final int entryNode;
    public final int maxDegree;

    CommonHeader(int version, int size, int dimension, int entryNode, int maxDegree) {
        this.version = version;
        this.size = size;
        this.dimension = dimension;
        this.entryNode = entryNode;
        this.maxDegree = maxDegree;
    }

    void write(DataOutput out) throws IOException {
        if (version >= 3) {
            out.writeInt(version);
        }
        out.writeInt(size);
        out.writeInt(dimension);
        out.writeInt(entryNode);
        out.writeInt(maxDegree);
    }

    static CommonHeader load(RandomAccessReader reader) throws IOException {
        int version = reader.readInt();
        int size = reader.readInt();
        int dimension = reader.readInt();
        int entryNode = reader.readInt();
        int maxDegree = reader.readInt();

        return new CommonHeader(version, size, dimension, entryNode, maxDegree);
    }

    static int size() {
        return 5 * Integer.BYTES;
    }
}
