/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.function.Supplier;
import java.util.logging.Logger;

/**
 * Provides random access to vectors by dense ordinal. This interface is used by graph-based
 * implementations of KNN search.
 */
public interface RandomAccessVectorValues {
    Logger LOG = Logger.getLogger(RandomAccessVectorValues.class.getName());

    /**
     * Return the number of vector values.
     * <p>
     * All copies of a given RAVV should have the same size.  Typically this is achieved by either
     * (1) implementing a threadsafe, un-shared RAVV, where `copy` returns `this`, or
     * (2) implementing a fixed-size RAVV.
     */
    int size();

    /** Return the dimension of the returned vector values */
    int dimension();

    /**
     * Return the vector value indexed at the given ordinal.
     *
     * <p>For performance, implementations are free to re-use the same object across invocations.
     * That is, you will get back the same VectorFloat&lt;?&gt;
     * reference (for instance) for every requested ordinal. If you want to use those values across
     * calls, you should make a copy.
     *
     * @param nodeId a valid ordinal, &ge; 0 and &lt; {@link #size()}.
     */
    VectorFloat<?> getVector(int nodeId);

    @Deprecated
    default VectorFloat<?> vectorValue(int targetOrd) {
        return getVector(targetOrd);
    }

    /**
     * Retrieve the vector associated with a given node, and store it in the destination vector at the given offset.
     * @param node the node to retrieve
     * @param destinationVector the vector to store the result in
     * @param offset the offset in the destination vector to store the result
     */
    default void getVectorInto(int node, VectorFloat<?> destinationVector, int offset) {
        destinationVector.copyFrom(getVector(node), 0, offset, dimension());
    }

    /**
     * @return true iff the vector returned by `getVector` is shared.  A shared vector will
     * only be valid until the next call to getVector overwrites it.
     */
    boolean isValueShared();

    /**
     * Creates a new copy of this {@link RandomAccessVectorValues}. This is helpful when you need to
     * access different values at once, to avoid overwriting the underlying float vector returned by
     * a shared {@link RandomAccessVectorValues#getVector}.
     * <p>
     * Un-shared implementations may simply return `this`.
     */
    RandomAccessVectorValues copy();

    /**
     * Returns a supplier of thread-local copies of the RAVV.
     */
    default Supplier<RandomAccessVectorValues> threadLocalSupplier() {
        if (!isValueShared()) {
            return () -> this;
        }

        if (this instanceof AutoCloseable) {
            LOG.warning("RAVV is shared and implements AutoCloseable; threadLocalSupplier() may lead to leaks");
        }
        var tl = ExplicitThreadLocal.withInitial(this::copy);
        return tl::get;
    }
}
