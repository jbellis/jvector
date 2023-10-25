package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutput;
import java.io.IOException;

public interface CompressedVectors extends Accountable {
    void write(DataOutput out) throws IOException;

    NeighborSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction);
}
