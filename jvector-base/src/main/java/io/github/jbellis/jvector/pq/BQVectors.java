package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.util.RamUsageEstimator;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

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
        // pq codebooks
        bq.write(out);

        // compressed vectors
        out.writeInt(compressedVectors.length);
        if (compressedVectors.length <= 0) {
            return;
        }
        out.writeInt(compressedVectors[0].length);
        for (var v : compressedVectors) {
            for (int i = 0; i < v.length; i++) {
                out.writeLong(v[i]);
            }
        }
    }

    @Override
    public NeighborSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        var qBQ = bq.encode(q);
        int bitLength = qBQ.length * Long.SIZE;
        return node2 -> {
            var vBQ = compressedVectors[node2];
            return 1 - (float) VectorUtil.hammingDistance(qBQ, vBQ) / bitLength;
        };
    }

    public long[] get(int i) {
        return compressedVectors[i];
    }

    @Override
    public long ramBytesUsed() {
        return compressedVectors.length * RamUsageEstimator.sizeOf(compressedVectors[0]);
    }
}
