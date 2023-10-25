package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;

public class BQVectors implements CompressedVectors {
    private final BinaryQuantization bq;
    private final long[][] compressedVectors;

    public BQVectors(BinaryQuantization bq, long[][] compressedVectors) {
        this.bq = bq;
        this.compressedVectors = compressedVectors;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        // TODO
    }

    @Override
    public NeighborSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        var qBQ = bq.encode(q);
        int bitLength = qBQ.length * Long.SIZE;
        return new NeighborSimilarity.ApproximateScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                var vBQ = compressedVectors[node2];
                int hammingDistance = 0;
                for (int i = 0; i < vBQ.length; i++) {
                    hammingDistance += Long.bitCount(vBQ[i] ^ qBQ[i]);
                }
                return 1 - (float) hammingDistance / bitLength;
            }
        };
    }

    @Override
    public long ramBytesUsed() {
        // TODO
        return 0;
    }
}
