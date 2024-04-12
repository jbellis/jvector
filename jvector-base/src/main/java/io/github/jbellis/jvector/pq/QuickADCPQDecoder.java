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

import io.github.jbellis.jvector.graph.disk.FusedADCNeighbors;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use Quick(er) ADC-style transposed vectors fused into a graph.
 */
public abstract class QuickADCPQDecoder implements ScoreFunction.ApproximateScoreFunction {
    protected final ProductQuantization pq;
    protected final VectorFloat<?> query;
    protected final ExactScoreFunction esf;

    protected QuickADCPQDecoder(ProductQuantization pq, VectorFloat<?> query, ExactScoreFunction esf) {
        this.pq = pq;
        this.query = query;
        this.esf = esf;
    }

    protected static abstract class CachingDecoder extends QuickADCPQDecoder {
        protected final VectorFloat<?> partialSums;
        protected CachingDecoder(ProductQuantization pq, VectorFloat<?> query, VectorSimilarityFunction vsf, ExactScoreFunction esf) {
            super(pq, query, esf);
            partialSums = pq.reusablePartialSums();

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
    }

     static class DotProductDecoder extends CachingDecoder {
        private final VectorFloat<?> results;
        private final FusedADCNeighbors neighbors;

        public DotProductDecoder(FusedADCNeighbors neighbors, ProductQuantization pq, VectorFloat<?> query, VectorFloat<?> results, ExactScoreFunction esf) {
            super(pq, query, VectorSimilarityFunction.DOT_PRODUCT, esf);
            this.neighbors = neighbors;
            this.results = results;
        }

        @Override
        public float similarityTo(int node2) {
            return esf.similarityTo(node2);
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            var permutedNodes = neighbors.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, pq.compressedVectorSize(), partialSums, results, VectorSimilarityFunction.DOT_PRODUCT);
            return results;
        }

        @Override
        public boolean supportsEdgeLoadingSimilarity() {
            return true;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        private final FusedADCNeighbors neighbors;
        private final VectorFloat<?> results;

        public EuclideanDecoder(FusedADCNeighbors neighbors, ProductQuantization pq, VectorFloat<?> query, VectorFloat<?> results, ExactScoreFunction esf) {
            super(pq, query, VectorSimilarityFunction.EUCLIDEAN, esf);
            this.neighbors = neighbors;
            this.results = results;
        }

        @Override
        public float similarityTo(int node2) {
            return esf.similarityTo(node2);
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            var permutedNodes = neighbors.getPackedNeighbors(origin);
            results.zero();
            VectorUtil.bulkShuffleSimilarity(permutedNodes, pq.compressedVectorSize(), partialSums, results, VectorSimilarityFunction.EUCLIDEAN);
            return results;
        }

        @Override
        public boolean supportsEdgeLoadingSimilarity() {
            return true;
        }
    }

    public static QuickADCPQDecoder newDecoder(FusedADCNeighbors neighbors, ProductQuantization pq, VectorFloat<?> query,
                                               VectorFloat<?> results, VectorSimilarityFunction similarityFunction, ExactScoreFunction esf) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return new DotProductDecoder(neighbors, pq, query, results, esf);
            case EUCLIDEAN:
                return new EuclideanDecoder(neighbors, pq, query, results, esf);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }
}
