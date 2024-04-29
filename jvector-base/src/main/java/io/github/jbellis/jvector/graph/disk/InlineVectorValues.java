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
import io.github.jbellis.jvector.util.ExceptionUtils;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * A RandomAccessVectorValues that knows how to load full-resolution vectors from an index that is in the process
 * of being written.
 */
public class InlineVectorValues implements RandomAccessVectorValues, Closeable {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final int dimension;
    private final OnDiskGraphIndexWriter writer;
    private final ExplicitThreadLocal<FeatureSource> sources;

    public InlineVectorValues(int dimension, OnDiskGraphIndexWriter writer) {
        this.dimension = dimension;
        this.writer = writer;
        sources = ExplicitThreadLocal.withInitial(writer::getFeatureSource);
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
        var v = new float[dimension];
        try {
            var reader = sources.get().inlineReaderForNode(ordinal, FeatureId.INLINE_VECTORS);
            reader.readFully(v);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return vts.createFloatVector(v);
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
        try {
            sources.close();
        } catch (Exception e) {
            ExceptionUtils.throwIoException(e);
        }
    }
}
