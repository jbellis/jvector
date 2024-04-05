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
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;

/**
 * Implements the storage of full-resolution vectors inline into an OnDiskGraphIndex. These can be used for exact scoring.
 */
class InlineVectors implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final int dimension;

    InlineVectors(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public int inlineSize() {
        return dimension * Float.BYTES;
    }

    static InlineVectors load(CommonHeader header, RandomAccessReader reader) {
        return new InlineVectors(header.dimension);
    }

    @Override
    public void writeHeader(DataOutput out) {
        // common header contains dimension, which is sufficient
    }

    FeatureWriter asWriter(RandomAccessVectorValues ravv) {
        return new FeatureWriter() {
            @Override
            public int inlineSize() {
                return InlineVectors.this.inlineSize();
            }

            @Override
            public void writeHeader(DataOutput out) throws IOException {
                InlineVectors.this.writeHeader(out);
            }

            @Override
            public void writeInline(int ordinal, DataOutput out) throws IOException {
                vectorTypeSupport.writeFloatVector(out, ravv.getVector(ordinal));
            }
        };
    }
}
