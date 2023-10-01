package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

/**
 * Performs similarity comparisons with compressed vectors without decoding them
 */
abstract class CompressedDecoder implements NeighborSimilarity.ApproximateScoreFunction {
    protected final CompressedVectors cv;

    protected CompressedDecoder(CompressedVectors cv) {
        this.cv = cv;
    }

    protected static abstract class CachingDecoder extends CompressedDecoder {
        protected final float[] partialSums;

        protected CachingDecoder(CompressedVectors cv, float[] query, VectorSimilarityFunction vsf) {
            super(cv);
            var pq = this.cv.pq;
            partialSums = cv.reusablePartialSums();

            float[] center = pq.getCenter();
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            for (var i = 0; i < pq.getSubspaceCount(); i++) {
                int offset = pq.subvectorSizesAndOffsets[i][1];
                int baseOffset = i * ProductQuantization.CLUSTERS;
                for (var j = 0; j < ProductQuantization.CLUSTERS; j++) {
                    float[] centroidSubvector = pq.codebooks[i][j];
                    switch (vsf) {
                        case DOT_PRODUCT:
                            partialSums[baseOffset + j] = VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                            break;
                        case EUCLIDEAN:
                            partialSums[baseOffset + j] = VectorUtil.squareDistance(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
                    }
                }
            }
        }

        protected float decodedSimilarity(byte[] encoded) {
            return VectorUtil.assembleAndSum(partialSums, ProductQuantization.CLUSTERS, encoded);
        }
    }

    static class DotProductDecoder extends CachingDecoder {
        public DotProductDecoder(CompressedVectors cv, float[] query) {
            super(cv, query, VectorSimilarityFunction.DOT_PRODUCT);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(cv.get(node2))) / 2;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(CompressedVectors cv, float[] query) {
            super(cv, query, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(cv.get(node2)));
        }
    }

    static class CosineDecoder extends CompressedDecoder {
        protected final float[] partialSums;
        protected final float[] aMagnitude;
        protected final float bMagnitude;

        public CosineDecoder(CompressedVectors cv, float[] query) {
            super(cv);
            var pq = this.cv.pq;

            // Compute and cache partial sums and magnitudes for query vector
            partialSums = cv.reusablePartialSums();
            aMagnitude = cv.reusablePartialMagnitudes();
            float bMagSum = 0.0f;

            float[] center = pq.getCenter();
            float[] centeredQuery = center == null ? query : VectorUtil.sub(query, center);

            for (int m = 0; m < pq.getSubspaceCount(); ++m) {
                int offset = pq.subvectorSizesAndOffsets[m][1];
                for (int j = 0; j < ProductQuantization.CLUSTERS; ++j) {
                    float[] centroidSubvector = pq.codebooks[m][j];
                    partialSums[(m * ProductQuantization.CLUSTERS) + j] = VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                    aMagnitude[(m * ProductQuantization.CLUSTERS) + j] = VectorUtil.dotProduct(centroidSubvector, 0, centroidSubvector, 0, centroidSubvector.length);
                }

                bMagSum += VectorUtil.dotProduct(centeredQuery, offset, centeredQuery, offset, pq.subvectorSizesAndOffsets[m][0]);
            }

            this.bMagnitude = bMagSum;
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedCosine(node2)) / 2;
        }

        protected float decodedCosine(int node2) {
            float sum = 0.0f;
            float aMag = 0.0f;

            byte[] encoded = cv.get(node2);

            for (int m = 0; m < encoded.length; ++m) {
                int centroidIndex = Byte.toUnsignedInt(encoded[m]);
                sum += partialSums[(m * ProductQuantization.CLUSTERS) + centroidIndex];
                aMag += aMagnitude[(m * ProductQuantization.CLUSTERS) + centroidIndex];
            }

            return (float) (sum / Math.sqrt(aMag * bMagnitude));
        }
    }
}
