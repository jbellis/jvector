package io.github.jbellis.jvector.pq;

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

/**
 * Interface for vector compression.  T is the encoded (compressed) vector type;
 * it will be an array type.
 */
public interface VectorCompressor<T> {
    T[] encodeAll(List<float[]> vectors);

    T encode(float[] v);

    void write(DataOutput out) throws IOException;

    /**
     * @param quantizedVectors must match the type T for this VectorCompression, but
     *                         it is declared as Object because we want callers to be able to use this
     *                         without committing to a specific type T.
     */
    CompressedVectors createCompressedVectors(Object[] quantizedVectors);
}
