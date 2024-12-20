package io.github.jbellis.jvector.pq;

public interface  MutableCompressedVectors<T> extends CompressedVectors {
    /**
     * Encode the given vector and set it at the given ordinal. Done without unnecessary copying.
     *
     * It's the caller's responsibility to ensure there are no "holes" in the ordinals that are
     * neither encoded nor set to zero.
     *
     * @param ordinal the ordinal to set
     * @param vector the vector to encode and set
     */
    void encodeAndSet(int ordinal, T vector);

    /**
     * Set the vector at the given ordinal to zero.
     *
     * It's the caller's responsibility to ensure there are no "holes" in the ordinals that are
     * neither encoded nor set to zero.
     *
     * @param ordinal the ordinal to set
     */
    void setZero(int ordinal);
}
