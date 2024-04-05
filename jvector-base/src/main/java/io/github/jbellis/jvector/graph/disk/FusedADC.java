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
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.pq.QuickADCPQDecoder;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Implements Quick ADC-style scoring by fusing PQ-encoded neighbors into an OnDiskGraphIndex.
 */
class FusedADC implements Feature {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ProductQuantization pq;
    private final int maxDegree;
    private final ThreadLocal<VectorFloat<?>> reusableResults;

    FusedADC(int maxDegree, ProductQuantization pq) {
        this.maxDegree = maxDegree;
        this.pq = pq;
        this.reusableResults = ThreadLocal.withInitial(() -> OnDiskGraphIndex.vectorTypeSupport.createFloatVector(maxDegree));
    }

    @Override
    public int inlineSize() {
        return pq.getCompressedSize() * maxDegree;
    }

    static FusedADC load(CommonHeader header, RandomAccessReader reader) {
        try {
            return new FusedADC(header.maxDegree, ProductQuantization.load(reader));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> queryVector, VectorSimilarityFunction vsf, OnDiskGraphIndex.View view, ScoreFunction.ExactScoreFunction esf) {
        var neighbors = new PackedNeighborsFused(view);
        return QuickADCPQDecoder.newDecoder(neighbors, pq, queryVector, reusableResults.get(), vsf, esf);
    }

    @Override
    public void writeHeader(DataOutput out) throws IOException {
        pq.write(out);
    }

    FeatureWriter asWriter(GraphIndex.View view, PQVectors pqVectors, int maxDegree) {
        ByteSequence<?> compressedNeighbors = vectorTypeSupport.createByteSequence(pqVectors.getCompressedSize() * maxDegree);
        return new FeatureWriter() {
            @Override
            public int inlineSize() {
                return FusedADC.this.inlineSize();
            }

            @Override
            public void writeHeader(DataOutput out) throws IOException {
                FusedADC.this.writeHeader(out);
            }

            @Override
            public void writeInline(int ordinal, DataOutput out) throws IOException {
                var neighbors = view.getNeighborsIterator(ordinal);
                int n = 0;
                var neighborSize = neighbors.size();
                compressedNeighbors.zero(); // TODO: make more efficient
                for (; n < neighborSize; n++) {
                    var compressed = pqVectors.get(neighbors.nextInt());
                    for (int j = 0; j < pqVectors.getCompressedSize(); j++) {
                        compressedNeighbors.set(j * maxDegree + n, compressed.get(j));
                    }
                }

                vectorTypeSupport.writeByteSequence(out, compressedNeighbors);
            }
        };
    }

    private class PackedNeighborsFused implements FusedADCNeighbors {
        private final OnDiskGraphIndex.View view;
        private final ByteSequence<?> neighbors;

        public PackedNeighborsFused(OnDiskGraphIndex.View view) {
            this.view = view;
            this.neighbors = OnDiskGraphIndex.vectorTypeSupport.createByteSequence(pq.getCompressedSize() * maxDegree);
        }

        @Override
        public ByteSequence<?> getPackedNeighbors(int node) {
            var reader = view.inlineReaderForNode(node, FeatureId.FUSED_ADC);
            try {
                OnDiskGraphIndex.vectorTypeSupport.readByteSequence(reader, neighbors);
                return neighbors;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
