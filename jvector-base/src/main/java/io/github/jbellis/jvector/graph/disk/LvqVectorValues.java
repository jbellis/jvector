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

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.Closeable;
import java.io.IOException;

/**
 * A RandomAccessVectorValues that knows how to decode LVQ vectors from an index that is in the process
 * of being written.
 * <p>
 * For doing a single comparison,
 * decoding to a VectorFloat as this does is slower than going through LAVQ.scoreFunctionFrom,
 * but for repeated comparisons as done in the diversity scoring, caching the
 * VectorFloat returned by this will be faster.
 */
public class LvqVectorValues implements RandomAccessVectorValues, Closeable {

    private final int dimension;
    private final OnDiskGraphIndexWriter writer;
    private final ExplicitThreadLocal<LVQPackedVectors> sources;

    public LvqVectorValues(int dimension, LVQ lvq, OnDiskGraphIndexWriter writer) {
        this.dimension = dimension;
        this.writer = writer;
        sources = ExplicitThreadLocal.withInitial(() -> new CloseablePackedVectors(lvq.createPackedVectors(writer.getFeatureSource())));
    }

    @Override
    public int size() {
        return writer.getMaxOrdinal();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    public VectorFloat<?> getVector(int ordinal) {
        var packed = sources.get().getPackedVector(ordinal);
        return packed.decode();
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public RandomAccessVectorValues copy() {
        return this;
    }

    @Override
    public void close() throws IOException {
        sources.close();
    }

    private static class CloseablePackedVectors implements LVQPackedVectors, Closeable {
        private final LVQ.PackedVectors raw;

        public CloseablePackedVectors(LVQ.PackedVectors raw) {
            this.raw = raw;
        }

        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int ordinal) {
            return raw.getPackedVector(ordinal);
        }

        @Override
        public void close() throws IOException {
            raw.source.close();
        }
    }
}
