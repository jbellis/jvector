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
import io.github.jbellis.jvector.pq.NVQScorer;
import io.github.jbellis.jvector.pq.NVQVectors;
import io.github.jbellis.jvector.pq.NVQuantization;
import io.github.jbellis.jvector.pq.NVQuantization.QuantizedVector;
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
 * Implements the storage of NuVeQ vectors in an on-disk graph index.  These can be used for reranking.
 */

public class NVQ implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final NVQuantization nvq;
    // private final ExplicitThreadLocal<ByteSequence<?>> reusableBytes;

    public NVQ(NVQuantization nvq) {
        this.nvq = nvq;
        // this.reusableBytes = ExplicitThreadLocal.withInitial(() -> vectorTypeSupport.createByteSequence(nvq.compressedVectorSize()));
    }

    @Override
    public FeatureId id() {
        return FeatureId.NVQ;
    }

    @Override
    public int headerSize() {
        return nvq.compressorSize();
    }

    @Override
    public int inlineSize() { return nvq.compressedVectorSize();}

    public int dimension() {
        return nvq.globalMean.length();
    }

    static NVQ load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new NVQ(NVQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        nvq.write(out, OnDiskGraphIndex.CURRENT_VERSION);
    }

    @Override
    public void writeInline(DataOutput out, Feature.State state_) throws IOException {
        var state = (NVQ.State) state_;
        state.vector.write(out);
    }

    public static class State implements Feature.State {
        public final QuantizedVector vector;

        public State(QuantizedVector vector) {
            this.vector = vector;
        }
    }

    ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> queryVector,
                                       VectorSimilarityFunction vsf,
                                       FeatureSource source)
    {
        NVQScorer scorer = new NVQScorer(this.nvq);
        return scorer.scoreFunctionFrom(queryVector, vsf, createPackedVectors(source));
    }

    public PackedQuantizedVectors createPackedVectors(FeatureSource source) {
        return new PackedQuantizedVectors(source);
    }

    /*
     * Retrieves one NVQ quantized vector from disk.
     */
    public static class PackedQuantizedVectors implements NVQPackedVectors {
        final FeatureSource source;

        public PackedQuantizedVectors(FeatureSource source) {
            this.source = source;
        }

        @Override
        public QuantizedVector getQuantizedVector(int ordinal) {
            // TODO determine if ExplicitThreadLocal should be used with subvector byte sequences
            try {
                var reader = source.inlineReaderForNode(ordinal, FeatureId.NVQ);
                return QuantizedVector.load(reader);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
