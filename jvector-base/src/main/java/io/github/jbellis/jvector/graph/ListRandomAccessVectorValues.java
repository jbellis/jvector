/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import java.util.List;

/**
 * A List-backed implementation of the {@link RandomAccessVectorValues} interface.
 * <p>
 * It is acceptable to provide this class to a GraphBuilder, and then continue
 * to add vectors to it as you add to the graph.
 * <p>
 * This will be as threadsafe as the provided List.
 */
public class ListRandomAccessVectorValues implements RandomAccessVectorValues<float[]> {

    private final List<float[]> vectors;
    private final int dimension;

    /**
     * Construct a new instance of {@link ListRandomAccessVectorValues}.
     *
     * @param vectors   a (potentially mutable) list of float vectors.
     * @param dimension the dimension of the vectors.
     */
    public ListRandomAccessVectorValues(List<float[]> vectors, int dimension) {
        this.vectors = vectors;
        this.dimension = dimension;
    }

    @Override
    public int size() {
        return vectors.size();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        return vectors.get(targetOrd);
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public ListRandomAccessVectorValues copy() {
        // our vectorValue is Unshared, but copy anyway in case the underlying List is not threadsafe
        return new ListRandomAccessVectorValues(List.copyOf(vectors), dimension);
    }
}
