package io.github.jbellis.jvector.vector.types;

import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorEncoding;

public interface VectorType<T extends Number, D> extends Accountable {

    VectorEncoding type();

    /**
     * @return entire vector
     */
    D get();

    int length();

    default int offset(int i) {
        return i;
    }

    VectorType<T, D> copy();
}
