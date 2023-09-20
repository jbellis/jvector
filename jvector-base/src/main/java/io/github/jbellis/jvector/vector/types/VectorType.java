package io.github.jbellis.jvector.vector.types;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorEncoding;

public interface VectorType<T extends Number, D> extends Accountable {

    VectorEncoding type();

    /**
     * @return entire vector
     */
    D get();

    T get(int n);

    void set(int n, T value);

    int length();

    int offset();

    VectorType<T, D> copy();
}
