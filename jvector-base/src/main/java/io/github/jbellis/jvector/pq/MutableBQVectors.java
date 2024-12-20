package io.github.jbellis.jvector.pq;

public class MutableBQVectors extends BQVectors implements MutableCompressedVectors<long[]> {
    /**
     * Construct a mutable BQVectors instance with the given BinaryQuantization and maximum number of vectors
     * that will be stored in this instance.
     * @param bq the BinaryQuantization to use
     * @param maximumVectorCount the maximum number of vectors that will be stored in this instance
     */
    public MutableBQVectors(BinaryQuantization bq, int maximumVectorCount) {
        super(bq);
        this.compressedVectors = new long[maximumVectorCount][];
        this.vectorCount = 0;
    }

    @Override
    public void encodeAndSet(int ordinal, long[] vector) {
        compressedVectors[ordinal] = vector;
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }

    @Override
    public void setZero(int ordinal) {
        compressedVectors[ordinal] = new long[bq.compressedVectorSize()];
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }
}
