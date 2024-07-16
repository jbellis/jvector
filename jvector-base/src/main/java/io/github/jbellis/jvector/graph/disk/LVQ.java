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
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements the storage of LVQ-quantized vectors in an on-disk graph index. These can be used for reranking.
 */
public class LVQ implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final LocallyAdaptiveVectorQuantization lvq;
    private final ExplicitThreadLocal<ByteSequence<?>> reusableBytes;

    public LVQ(LocallyAdaptiveVectorQuantization lvq) {
        this.lvq = lvq;
        this.reusableBytes = ExplicitThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(lvq.compressedVectorSize() - 2 * Float.BYTES));
    }

    @Override
    public FeatureId id() {
        return FeatureId.LVQ;
    }

    @Override
    public int headerSize() {
        return lvq.compressorSize();
    }

    @Override
    public int inlineSize() {
        return lvq.compressedVectorSize();
    }

    public int dimension() {
        return lvq.globalMean.length();
    }

    static LVQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new LVQ(LocallyAdaptiveVectorQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        lvq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeInline(DataOutput out, Feature.State state_) throws IOException {
        var state = (LVQ.State) state_;
        state.vector.writePacked(out);
    }

    public static class State implements Feature.State {
        public final LocallyAdaptiveVectorQuantization.QuantizedVector vector;

        public State(LocallyAdaptiveVectorQuantization.QuantizedVector vector) {
            this.vector = vector;
        }
    }

    ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector,
                                       VectorSimilarityFunction vsf,
                                       FeatureSource source)
    {
        return lvq.scoreFunctionFrom(queryVector, vsf, createPackedVectors(source));
    }

    public PackedVectors createPackedVectors(FeatureSource source) {
        return new PackedVectors(source);
    }

    public class PackedVectors implements LVQPackedVectors {
        final FeatureSource source;

        public PackedVectors(FeatureSource source) {
            this.source = source;
        }

        @Override
        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int ordinal) {
            try {
                var reader = source.inlineReaderForNode(ordinal, FeatureId.LVQ);
                var bias = reader.readFloat();
                var scale = reader.readFloat();
                var tlBytes = reusableBytes.get();
                // reduce the size by 2 floats read as bias/scale
                OnDiskGraphIndex.vectorTypeSupport.readByteSequence(reader, tlBytes);
                return new LocallyAdaptiveVectorQuantization.PackedVector(tlBytes, bias, scale);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
