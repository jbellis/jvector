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

import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DataSetCreator {
    /**
     * Creates a 2D grid of vectors, query vectors, and ground truth data for a given grid width.
     *
     * @param nPoints  Number of points in the grid.
     * @param nQueries Number of query vectors to generate.
     * @param topK     The number of closest points to find for each query vector.
     * @return DataSet containing base vectors, query vectors, and ground truth sets.
     */
    public static DataSet create2DGrid(int nPoints, int nQueries, int topK) {
        // generate the grid
        int gridWidth = (int) Math.sqrt(nPoints);
        var baseVectors = new ArrayList<float[]>();
        for (int x = 0; x < gridWidth; x++) {
            for (int y = 0; y < gridWidth; y++) {
                baseVectors.add(new float[] {x, y});
            }
        }

        // Create query vectors and compute ground truth
        var queries = PhysicalCoreExecutor.instance.submit(() -> IntStream.range(0, nQueries).parallel().mapToObj(i -> {
            var R = ThreadLocalRandom.current();
            float[] q = new float[]{gridWidth * R.nextFloat(), gridWidth * R.nextFloat()};

            // Compute the ground truth within a bounding box around the query point
            Set<Integer> gt = IntStream.range(0, baseVectors.size())
                    .filter(j -> {
                        float[] v = baseVectors.get(j);
                        return v[0] >= q[0] - topK && v[0] <= q[0] + topK && v[1] >= q[1] - topK && v[1] <= q[1] + topK;
                    })
                    .boxed() // allows sorting with custom comparator
                    .sorted(Comparator.comparingDouble((Integer j) -> VectorSimilarityFunction.EUCLIDEAN.compare(q, baseVectors.get(j))).reversed())
                    .limit(topK)
                    .collect(Collectors.toSet());

            return new AbstractMap.SimpleEntry<>(q, gt);
        }).collect(Collectors.toConcurrentMap(Map.Entry::getKey, Map.Entry::getValue)).entrySet());
        var queryVectors = queries.stream().map(Map.Entry::getKey).collect(Collectors.toList());
        var groundTruth = queries.stream().map(Map.Entry::getValue).collect(Collectors.toList());

        String name = "2D" + gridWidth;
        return new DataSet(name, VectorSimilarityFunction.EUCLIDEAN, baseVectors, queryVectors, groundTruth);
    }
}
