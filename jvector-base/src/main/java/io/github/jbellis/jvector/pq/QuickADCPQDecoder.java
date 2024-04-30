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
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Performs similarity comparisons with compressed vectors without decoding them.
 * These decoders use Quick(er) ADC-style transposed vectors fused into a graph.
 */
// TODO remove one layer of nested inheritance (merge CachingDecoder into QuickADCPQDecoder)
// (unless we're going to follow the structure of PQDecoder and add a CosineDecoder with a different structure)
public abstract class QuickADCPQDecoder implements ScoreFunction.ApproximateScoreFunction {
    protected final ProductQuantization pq;
    protected final VectorFloat<?> query;
    protected final ExactScoreFunction esf;

    protected QuickADCPQDecoder(ProductQuantization pq, VectorFloat<?> query, ExactScoreFunction esf) {
        this.pq = pq;
        this.query = query;
        this.esf = esf;
    }

    // Implements section 3.4 of "Quicker ADC : Unlocking the Hidden Potential of Product Quantization with SIMD"
    // The main difference is that since our graph structure rapidly converges towards the best results,
    // we don't need to scan K values to have enough confidence that our worstDistance bound is reasonable.
    protected static abstract class CachingDecoder extends QuickADCPQDecoder {
        // connected to the Graph View by caller
        protected final FusedADCNeighbors neighbors;
        // caller passes this to us for re-use across calls
        protected final VectorFloat<?> results;
        // decoder state
        protected final VectorFloat<?> partialSums;
        protected final ByteSequence<?> partialQuantizedSums;
        protected final VectorFloat<?> partialBestDistances;
        private final VectorSimilarityFunction vsf;
        protected final float bestDistance;
        protected final int invocationThreshold;
        protected float worstDistance;
        protected int invocations;
        protected boolean supportsQuantizedSimilarity;
        protected float delta;

        protected CachingDecoder(FusedADCNeighbors neighbors, VectorFloat<?> results, ProductQuantization pq, VectorFloat<?> query, int invocationThreshold, VectorSimilarityFunction vsf, ExactScoreFunction esf) {
            super(pq, query, esf);
            this.neighbors = neighbors;
            this.results = results;
            this.vsf = vsf;
            this.invocationThreshold = invocationThreshold;

            // compute partialSums, partialBestDistances, and bestDistance from the codebooks
            partialSums = pq.reusablePartialSums();
            partialBestDistances = pq.reusablePartialBestDistances();
            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int size = pq.subvectorSizesAndOffsets[i][0];
                var codebook = pq.codebooks[i];
                VectorUtil.calculatePartialSums(codebook, i, size, pq.getClusterCount(), centeredQuery, offset, vsf, partialSums, partialBestDistances);
            }
            bestDistance = VectorUtil.sum(partialBestDistances);

            // these will be computed by edgeLoadingSimilarityTo as we search
            partialQuantizedSums = pq.reusablePartialQuantizedSums();
            delta = 0;
            worstDistance = 0;
            // internal state for edgeLoadingSimilarityTo
            invocations = 0;
            supportsQuantizedSimilarity = false;
        }

        @Override
        public VectorFloat<?> edgeLoadingSimilarityTo(int origin) {
            var permutedNodes = neighbors.getPackedNeighbors(origin);
            results.zero();

            if (supportsQuantizedSimilarity) {
                // we have seen enough data to compute `delta`, so take the fast path using the permuted nodes
                VectorUtil.bulkShuffleQuantizedSimilarity(permutedNodes, pq.compressedVectorSize(), partialQuantizedSums, delta, bestDistance, results, vsf);
                return results;
            }

            // we have not yet computed worstDistance or delta, so we need to assemble the results manually
            // from the PQ codebooks
            var nodeCount = results.length();
            for (int i = 0; i < pq.getSubspaceCount(); i++) {
                for (int j = 0; j < nodeCount; j++) {
                    results.set(j, results.get(j) + partialSums.get(i * pq.getClusterCount() + Byte.toUnsignedInt(permutedNodes.get(i * nodeCount + j))));
                }
            }
            // update worstDistance from our new set of results
            for (int i = 0; i < nodeCount; i++) {
                var result = results.get(i);
                invocations++;
                worstDistance = Math.min(worstDistance, result);
                results.set(i, distanceToScore(result));
            }
            // once we have enough data, set up delta and partialQuantizedSums for the fast path
            if (invocations >= invocationThreshold) {
                delta = (worstDistance - bestDistance) / 65535;
                VectorUtil.quantizePartialSums(delta, partialSums, partialBestDistances, partialQuantizedSums);
                supportsQuantizedSimilarity = true;
            }

            return results;
        }

        @Override
        public boolean supportsEdgeLoadingSimilarity() {
            return true;
        }

        @Override
        public float similarityTo(int node2) {
            return esf.similarityTo(node2);
        }

        protected abstract float distanceToScore(float distance);
    }

    static class DotProductDecoder extends CachingDecoder {
        public DotProductDecoder(FusedADCNeighbors neighbors, ProductQuantization pq, VectorFloat<?> query, VectorFloat<?> results, ExactScoreFunction esf) {
            super(neighbors, results, pq, query, neighbors.maxDegree(), VectorSimilarityFunction.DOT_PRODUCT, esf);
            worstDistance = Float.MAX_VALUE;
        }

        @Override
        protected float distanceToScore(float distance) {
            return (distance + 1) / 2;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(FusedADCNeighbors neighbors, ProductQuantization pq, VectorFloat<?> query, VectorFloat<?> results, ExactScoreFunction esf) {
            super(neighbors, results, pq, query, neighbors.maxDegree(), VectorSimilarityFunction.EUCLIDEAN, esf);
            worstDistance = Float.MIN_VALUE;
        }

        @Override
        protected float distanceToScore(float distance) {
            return 1 / (1 + distance);
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
