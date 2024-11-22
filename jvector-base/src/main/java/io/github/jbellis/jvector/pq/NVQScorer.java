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

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public class NVQScorer {
    final NVQuantization nvq;

    /**
     * Initialize the NVQ Vectors with an initial List of vectors.  This list may be
     * mutated, but caller is responsible for thread safety issues when doing so.
     */
    public NVQScorer(NVQuantization nvq) {
        this.nvq = nvq;
    }

    public NVQScoreFunction scoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        switch (similarityFunction) {
            case DOT_PRODUCT:
                return dotProductScoreFunctionFor(query);
            case EUCLIDEAN:
                return euclideanScoreFunctionFor(query);
            case COSINE:
                return cosineScoreFunctionFor(query);
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }

    private NVQScoreFunction dotProductScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         * The dot product is calculated between the query and quantized sub-vectors as follows:
         *
         * <query, vector> \approx <query, scale * quantized + bias + globalMean>
         *                       = scale * <query, quantized> + bias <query, broadcast(1)> + <query, globalMean>
         *
         * where scale and bias are scalars.
         *
         * The following terms can be precomputed:
         *     queryGlobalBias = <query, globalMean>
         *     querySum = <query, broadcast(1)>
         */
        var queryGlobalBias = VectorUtil.dotProduct(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(query);

        var querySum = new float[querySubVectors.length];
        for (int i = 0; i < querySubVectors.length; i++) {
            querySum[i] = VectorUtil.sum(querySubVectors[i]);
        }

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                return vector2 -> {
                    float nvqDot = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        nvqDot += VectorUtil.nvqDotProduct8bit(querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias,
                                querySum[i]
                        );
                    }
                    // TODO This won't work without some kind of normalization.  Intend to scale [0, 1]
                    return (1 + nvqDot + queryGlobalBias) / 2;
                };
            case FOUR:
                for (VectorFloat<?> querySubVector : querySubVectors) {
                    VectorUtil.nvqShuffleQueryInPlace4bit(querySubVector);
                }

                return vector2 -> {
                    float nvqDot = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        nvqDot += VectorUtil.nvqDotProduct4bit(querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias,
                                querySum[i]
                        );
                    }
                    // TODO This won't work without some kind of normalization.  Intend to scale [0, 1]
                    return (1 + nvqDot + queryGlobalBias) / 2;
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    private NVQScoreFunction euclideanScoreFunctionFor(VectorFloat<?> query) {
        /* Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
         * first de-meaned by subtracting the global mean.
         *
         * The squared L2 distance is calculated between the query and quantized sub-vectors as follows:
         *
         * |query - vector|^2 \approx |query - (scale * quantized + bias + globalMean)|^2
         *                          = |(query - globalMean) - scale * quantized + bias|^2
         *
         * where scale and bias are scalars.
         *
         * The following term can be precomputed:
         *     shiftedQuery = query - globalMean
         */
        var shiftedQuery = VectorUtil.sub(query, this.nvq.globalMean);
        var querySubVectors = this.nvq.getSubVectors(shiftedQuery);

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                return vector2 -> {
                    float dist = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        dist += VectorUtil.nvqSquareL2Distance8bit(
                                querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias
                        );
                    }

                    return 1 / (1 + dist);
                };
            case FOUR:
                for (VectorFloat<?> querySubVector : querySubVectors) {
                    VectorUtil.nvqShuffleQueryInPlace4bit(querySubVector);
                }

                return vector2 -> {
                    float dist = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        dist += VectorUtil.nvqSquareL2Distance4bit(querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias
                        );
                    }

                    return 1 / (1 + dist);
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    private NVQScoreFunction cosineScoreFunctionFor(VectorFloat<?> query) {
        float queryNorm = (float) Math.sqrt(VectorUtil.dotProduct(query, query));
        var querySubVectors = this.nvq.getSubVectors(query);
        var meanSubVectors = this.nvq.getSubVectors(this.nvq.globalMean);

        switch (this.nvq.bitsPerDimension) {
            case EIGHT:
                return vector2 -> {
                    float cos = 0;
                    float squaredNormalization = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        var partialCosSim = VectorUtil.nvqCosine8bit(querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias,
                                meanSubVectors[i]);
                        cos += partialCosSim[0];
                        squaredNormalization += partialCosSim[1];
                    }
                    float cosine = (cos / queryNorm) / (float) Math.sqrt(squaredNormalization);

                    return (1 + cosine) / 2;
                };
            case FOUR:
                for (var i = 0; i < querySubVectors.length; i++) {
                    VectorUtil.nvqShuffleQueryInPlace4bit(querySubVectors[i]);
                    VectorUtil.nvqShuffleQueryInPlace4bit(meanSubVectors[i]);
                }

                return vector2 -> {
                    float dotProduct = 0;
                    float squaredNormalization = 0;
                    for (int i = 0; i < querySubVectors.length; i++) {
                        var svDB = vector2.subVectors[i];
                        var partialCosSim = VectorUtil.nvqCosine4bit(querySubVectors[i],
                                svDB.bytes, svDB.kumaraswamyA, svDB.kumaraswamyB,
                                svDB.kumaraswamyScale, svDB.kumaraswamyBias,
                                meanSubVectors[i]);
                        dotProduct += partialCosSim[0];
                        squaredNormalization += partialCosSim[1];
                    }
                    float cosine = (dotProduct / queryNorm) / (float) Math.sqrt(squaredNormalization);

                    return (1 + cosine) / 2;
                };
            default:
                throw new IllegalArgumentException("Unsupported bits per dimension " + this.nvq.bitsPerDimension);
        }
    }

    public interface NVQScoreFunction {
        /**
         * @return the similarity to another vector
         */
        float similarityTo(NVQuantization.QuantizedVector vector2);
    }
}
