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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.List;
import java.util.Set;

public class DataSet {
    public final String name;
    public final VectorSimilarityFunction similarityFunction;
    public final List<float[]> baseVectors;
    public final List<float[]> queryVectors;
    public final List<? extends Set<Integer>> groundTruth;

    public DataSet(String name, VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<? extends Set<Integer>> groundTruth) {
        if (baseVectors.isEmpty() || queryVectors.isEmpty() || groundTruth.isEmpty()) {
            throw new IllegalArgumentException("Base, query, and groundTruth vectors must not be empty");
        }
        if (baseVectors.get(0).length != queryVectors.get(0).length) {
            throw new IllegalArgumentException("Base and query vectors must have the same dimensionality");
        }
        if (queryVectors.size() != groundTruth.size()) {
            throw new IllegalArgumentException("Query and ground truth lists must be the same size");
        }

        this.name = name;
        this.similarityFunction = similarityFunction;
        this.baseVectors = baseVectors;
        this.queryVectors = queryVectors;
        this.groundTruth = groundTruth;
    }
}
