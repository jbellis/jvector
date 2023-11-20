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

import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DataSet {
    public final String name;
    public final VectorSimilarityFunction similarityFunction;
    public final List<float[]> baseVectors;
    public final List<float[]> queryVectors;
    public final List<? extends Set<Integer>> groundTruth;
    private RandomAccessVectorValues<float[]> baseRavv;

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

        System.out.format("%n%s: %d base and %d query vectors created, dimensions %d%n",
                name, baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);
    }

    /**
     * Return a dataset containing the given vectors, scrubbed free from zero vectors and normalized to unit length.
     * Note: This only scrubs and normalizes for dot product similarity.
     */
    public static DataSet getScrubbedDataSet(String pathStr, VectorSimilarityFunction similarityFunction,
                                             List<float[]> baseVectors, List<float[]> queryVectors, List<HashSet<Integer>> groundTruth) {
        List<float[]> scrubbedBaseVectors;
        List<float[]> scrubbedQueryVectors;
        List<HashSet<Integer>> gtSet;
        if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            // verify that vectors are normalized and sane.
            // this is necessary b/c NYT (and NW) dataset(s) contain(s) zero vectors (!)
            scrubbedBaseVectors = new ArrayList<>(baseVectors.size());
            scrubbedQueryVectors = new ArrayList<>(queryVectors.size());
            gtSet = new ArrayList<>(groundTruth.size());
            // remove zero vectors, noting that this will change the indexes of the ground truth answers
            Map<Integer, Integer> rawToScrubbed = new HashMap<>();
            {
                int j = 0;
                for (int i = 0; i < baseVectors.size(); i++) {
                    float[] v = baseVectors.get(i);
                    if (Math.abs(normOf(v)) > 1e-5) {
                        scrubbedBaseVectors.add(v);
                        rawToScrubbed.put(i, j++);
                    }
                }
            }
            for (int i = 0; i < queryVectors.size(); i++) {
                float[] v = queryVectors.get(i);
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedQueryVectors.add(v);
                    var gt = new HashSet<Integer>();
                    for (int j : groundTruth.get(i)) {
                        gt.add(rawToScrubbed.get(j));
                    }
                    gtSet.add(gt);
                }
            }
            // now that the zero vectors are removed, we can normalize
            if (Math.abs(normOf(baseVectors.get(0)) - 1.0) > 1e-5) {
                normalizeAll(scrubbedBaseVectors);
                normalizeAll(scrubbedQueryVectors);
            }
            assert scrubbedQueryVectors.size() == gtSet.size();
        } else {
            scrubbedBaseVectors = baseVectors;
            scrubbedQueryVectors = queryVectors;
            gtSet = groundTruth;
        }

        return new DataSet(pathStr, similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    public int getDimension() {
        return baseVectors.get(0).length;
    }

    public RandomAccessVectorValues<float[]> getBaseRavv() {
        if (baseRavv == null) {
            baseRavv = new ListRandomAccessVectorValues(baseVectors, getDimension());
        }
        return baseRavv;
    }
}
