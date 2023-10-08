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

abstract class AbstractMockVectorValues<T> implements RandomAccessVectorValues<T> {
    protected final int dimension;
    protected final T[] denseValues;
    protected final int numVectors;

    AbstractMockVectorValues(int dimension, T[] denseValues, int numVectors) {
        this.dimension = dimension;
        this.denseValues = denseValues;
        // used by tests that build a graph from bytes rather than floats
        this.numVectors = numVectors;
    }

    @Override
    public int size() {
        return numVectors;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public boolean isValueShared() {
        return false;
    }

    @Override
    public T vectorValue(int targetOrd) {
        return denseValues[targetOrd];
    }

    @Override
    public abstract AbstractMockVectorValues<T> copy();
}
