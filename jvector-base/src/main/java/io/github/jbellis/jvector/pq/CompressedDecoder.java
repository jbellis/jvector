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
        protected final float[][] cache;

        protected CachingDecoder(CompressedVectors cv, float[] query, VectorSimilarityFunction vsf) {
            super(cv);
            cache = computeQueryFragments(query, vsf);
        }

        public float[][] computeQueryFragments(float[] query, VectorSimilarityFunction vsf) {
            var pq = cv.pq;
            float[][] cache = new float[pq.getSubspaceCount()][];
            float[] centroid = pq.getCenter();
            var centeredQuery = centroid == null ? query : VectorUtil.sub(query, centroid);
            for (var i = 0; i < cache.length; i++) {
                cache[i] = new float[ProductQuantization.CLUSTERS];
                for (var j = 0; j < ProductQuantization.CLUSTERS; j++) {
                    int offset = pq.subvectorSizesAndOffsets[i][1];
                    float[] centroidSubvector = pq.codebooks[i][j];
                    switch (vsf) {
                        case DOT_PRODUCT:
                            cache[i][j] = VectorUtil.dotProduct(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                            break;
                        case EUCLIDEAN:
                            cache[i][j] = VectorUtil.squareDistance(centroidSubvector, 0, centeredQuery, offset, centroidSubvector.length);
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
                    }
                }
            }
            return cache;
        }

        protected static float decodedSimilarity(byte[] encoded, float[][] cache) {
            // combining cached fragments is the same for dot product and euclidean; cosine is not supported
            float sum = 0.0f;
            for (int m = 0; m < cache.length; ++m) {
                int centroidIndex = Byte.toUnsignedInt(encoded[m]);
                var cachedValue = cache[m][centroidIndex];
                sum += cachedValue;
            }
            return sum;
        }
    }

    static class DotProductDecoder extends CachingDecoder {
        public DotProductDecoder(CompressedVectors cv, float[] query) {
            super(cv, query, VectorSimilarityFunction.DOT_PRODUCT);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(cv.get(node2), cache)) / 2;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(CompressedVectors cv, float[] query) {
            super(cv, query, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(cv.get(node2), cache));
        }
    }

    static class CosineDecoder extends CompressedDecoder {
        private final float[] query;

        public CosineDecoder(CompressedVectors cv, float[] query) {
            super(cv);
            this.query = query;
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedCosine(node2)) / 2;
        }

        private float decodedCosine(int node2) {
            byte[] encoded = cv.get(node2);
            float sum = 0.0f;
            float aMagnitude = 0.0f;
            float bMagnitude = 0.0f;
            for (int m = 0; m < cv.pq.M; ++m) {
                int offset = cv.pq.subvectorSizesAndOffsets[m][1];
                int centroidIndex = Byte.toUnsignedInt(encoded[m]);
                float[] centroidSubvector = cv.pq.codebooks[m][centroidIndex];
                var length = centroidSubvector.length;
                sum += VectorUtil.dotProduct(centroidSubvector, 0, query, offset, length);
                aMagnitude += VectorUtil.dotProduct(centroidSubvector, 0, centroidSubvector, 0, length);
                bMagnitude +=  VectorUtil.dotProduct(query, offset, query, offset, length);
            }
            return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
        }
    }
}
