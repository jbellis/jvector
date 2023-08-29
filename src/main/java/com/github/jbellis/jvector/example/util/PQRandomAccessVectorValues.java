package com.github.jbellis.jvector.example.util;

import com.github.jbellis.jvector.disk.CompressedVectors;
import com.github.jbellis.jvector.graph.RandomAccessVectorValues;
import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.List;

/**
 * A PQ-backed implementation of the {@link RandomAccessVectorValues} interface.
 *
 * TODO this is a quick hack job and should be replaced, wrapping CV until full DiskANN
 */
public class PQRandomAccessVectorValues implements RandomAccessVectorValues<float[]> {
    private final CompressedVectors cv;

    public PQRandomAccessVectorValues(ProductQuantization pq, List<byte[]> vectors) {
        this.cv = new CompressedVectors(pq, vectors);
    }

    @Override
    public int size() {
        return cv.getCompressedVectors().size();
    }

    @Override
    public int dimension() {
        return cv.getPq().vectorDimension();
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        throw new UnsupportedOperationException();
    }

    public float decodedSimilarity(int targetOrd, float[] query, VectorSimilarityFunction similarityFunction) {
        return cv.decodedSimilarity(targetOrd, query, similarityFunction);
    }

    @Override
    public PQRandomAccessVectorValues copy() {
        return new PQRandomAccessVectorValues(cv.getPq(), cv.getCompressedVectors());
    }
}
