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
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements the storage of LVQ-quantized vectors in an on-disk graph index. These can be used for reranking.
 */
class LVQ implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final LocallyAdaptiveVectorQuantization lvq;
    private final int lvqVectorSize;

    LVQ(LocallyAdaptiveVectorQuantization lvq, int lvqVectorSize) {
        this.lvq = lvq;
        this.lvqVectorSize = lvqVectorSize;
    }

    @Override
    public int inlineSize() {
        return lvqVectorSize;
    }

    static LVQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            var lvqDimension = header.dimension % 64 == 0 ? header.dimension : (header.dimension / 64 + 1) * 64;
            return new LVQ(LocallyAdaptiveVectorQuantization.load(reader), lvqDimension + 2 * Float.BYTES);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        lvq.write(out);
    }

    FeatureWriter asWriter(LocallyAdaptiveVectorQuantization.QuantizedVector[] lvqVectors) {
        return new FeatureWriter() {
            @Override
            public int inlineSize() {
                return LVQ.this.inlineSize();
            }

            @Override
            public void writeHeader(DataOutput out) throws IOException {
                LVQ.this.writeHeader(out);
            }

            @Override
            public void writeInline(int ordinal, DataOutput out) throws IOException {
                lvqVectors[ordinal].writePacked(out);
            }
        };
    }

    ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, OnDiskGraphIndex.View view) {

        return lvq.scoreFunctionFrom(queryVector, vsf, new PackedVectors(view));
    }

    private class PackedVectors implements LVQPackedVectors {
        private final OnDiskGraphIndex.View view;

        public PackedVectors(OnDiskGraphIndex.View view) {
            this.view = view;
        }


        @Override
        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int ordinal) {
            var reader = view.inlineReaderForNode(ordinal, FeatureId.LVQ);
            try {
                var bias = reader.readFloat();
                var scale = reader.readFloat();
                var packed = OnDiskGraphIndex.vectorTypeSupport.readByteSequence(reader, lvqVectorSize);
                return new LocallyAdaptiveVectorQuantization.PackedVector(packed, bias, scale);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
