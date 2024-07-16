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
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements the storage of PQ vectors inline into an OnDiskGraphIndex. These can be used for reranking.
 */
public class InlinePQ implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;

    public InlinePQ(ProductQuantization pq) {
        this.pq = pq;
    }

    @Override
    public FeatureId id() {
        return FeatureId.INLINE_PQ;
    }

    @Override
    public int headerSize() {
        return pq.compressorSize();
    }

    public int inlineSize() {
        return pq.compressedVectorSize();
    }

    public int dimension() {
        return pq.originalDimension;
    }

    static InlinePQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new InlinePQ(ProductQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(DataOutput out) {
        try {
            pq.write(out);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeInline(DataOutput out, Feature.State state) throws IOException {
        vectorTypeSupport.writeByteSequence(out, ((State) state).encoded);
    }

    ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector,
                                                 VectorSimilarityFunction vsf,
                                                 FeatureSource source)
    {
        return rerankerFor(queryVector, vsf, new InlineSource(source));
    }

    public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, ProductQuantization.EncodedSource source) {
        var sf = pq.scoreFunctionFor(queryVector, vsf, source);
        return sf::similarityTo;
    }

    public static class State implements Feature.State {
        public final ByteSequence<?> encoded;

        public State(ByteSequence<?> encoded) {
            this.encoded = encoded;
        }
    }

    private class InlineSource implements ProductQuantization.EncodedSource {
        private final FeatureSource source;

        public InlineSource(FeatureSource source) {
            this.source = source;
        }

        @Override
        public ByteSequence<?> get(int ordinal) {
            try {
                var reader = source.inlineReaderForNode(ordinal, FeatureId.INLINE_PQ);
                return vectorTypeSupport.readByteSequence(reader, pq.compressedVectorSize());
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
