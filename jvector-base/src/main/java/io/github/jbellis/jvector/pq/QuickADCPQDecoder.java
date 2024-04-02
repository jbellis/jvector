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

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.disk.ADCView;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use Quick(er) ADC-style transposed vectors fused into a graph.
 */
public abstract class QuickADCPQDecoder implements ScoreFunction.ApproximateScoreFunction {
    protected final PQVectors pqv;

    protected QuickADCPQDecoder(PQVectors pqv) {
        this.pqv = pqv;
    }

    protected static abstract class CachingDecoder extends QuickADCPQDecoder {
        protected final VectorFloat<?> partialSums;
        protected CachingDecoder(PQVectors pqv, VectorFloat<?> query, VectorSimilarityFunction vsf) {
            super(pqv);
            partialSums = pqv.reusablePartialSums();
            var pq = this.pqv.pq;

            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                int baseOffset = i * pq.getClusterCount();
                var codebook = pq.codebooks[i];
                VectorUtil.calculatePartialSums(codebook, baseOffset, size, pq.getClusterCount(), centeredQuery, offset, vsf, partialSums);
            }
        }

        protected float decodedSimilarity(ByteSequence<?> encoded) {
            return VectorUtil.assembleAndSum(partialSums, pqv.pq.getClusterCount(), encoded);
        }
    }

     static class DotProductDecoder extends CachingDecoder {
        private final VectorFloat<?> results;
        private final ADCView view;

        public DotProductDecoder(ADCView view, VectorFloat<?> query) {
            super(view.getPQVectors(), query, VectorSimilarityFunction.DOT_PRODUCT);
            this.view = view;
            this.results = view.reusableResults();
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(pqv.get(node2))) / 2;
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            var permutedNodes = view.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, view.getPQVectors().getCompressedSize(), partialSums, results, VectorSimilarityFunction.DOT_PRODUCT);
            return results;
        }

        @Override
        public boolean supportsEdgeLoadingSimilarity() {
            return true;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        private final ADCView view;
        private final VectorFloat<?> results;


        public EuclideanDecoder(ADCView view, VectorFloat<?> query) {
            super(view.getPQVectors(), query, VectorSimilarityFunction.EUCLIDEAN);
            this.view = view;
            this.results = view.reusableResults();
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(pqv.get(node2)));
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            var permutedNodes = view.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, view.getPQVectors().getCompressedSize(), partialSums, results, VectorSimilarityFunction.EUCLIDEAN);
            return results;
        }

        @Override
        public boolean supportsEdgeLoadingSimilarity() {
            return true;
        }
    }

    public static QuickADCPQDecoder newDecoder(ADCView view, VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new DotProductDecoder(view, query);
            case EUCLIDEAN:
                return new EuclideanDecoder(view, query);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }
}
