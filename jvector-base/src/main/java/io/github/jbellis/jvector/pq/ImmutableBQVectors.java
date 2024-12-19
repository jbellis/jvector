package io.github.jbellis.jvector.pq;

public class ImmutableBQVectors extends BQVectors {
    public ImmutableBQVectors(BinaryQuantization bq, long[][] compressedVectors) {
        super(bq);
        this.compressedVectors = compressedVectors;
    }
}
