/*
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

package com.github.jbellis.jvector.example;

import com.github.jbellis.jvector.example.util.PQRandomAccessVectorValues;
import com.github.jbellis.jvector.graph.*;
import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;
import com.github.jbellis.jvector.vector.VectorUtil;
import io.jhdf.HdfFile;

import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    private static void testRecall(int M, int efConstruction, List<Boolean> pqOptions, List<Integer> efSearchOptions, DataSet ds, PQRandomAccessVectorValues pqvv)
    {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        var topK = ds.groundTruth.get(0).size();

        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.5f, 1.4f);
        var index = builder.build();
        long buildNanos = System.nanoTime() - start;

        int queryRuns = 10;
        for (int overquery : efSearchOptions) {
            for (boolean usePq : pqOptions) {
                start = System.nanoTime();
                var pqr = performQueries(ds, usePq ? pqvv : floatVectors, index, topK, topK * overquery, queryRuns);
                var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
                System.out.format("Index   M=%d ef=%d PQ=%b: top %d/%d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                        M, efConstruction, usePq, topK, overquery, recall, buildNanos / 1_000_000_000.0, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
            }
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record ResultSummary(int topKFound, int nodesVisited) { }

    private static long topKCorrect(int topK, int[] resultNodes, Set<Integer> gt) {
        int count = Math.min(resultNodes.length, topK);
        // stream the first count results into a Set
        var resultSet = Arrays.stream(resultNodes, 0, count)
                .boxed()
                .collect(Collectors.toSet());
        assert resultSet.size() == count : String.format("%s duplicate results out of %s", count - resultSet.size(), count);
        return resultSet.stream().filter(gt::contains).count();
    }

    private static ResultSummary performQueries(DataSet ds, RandomAccessVectorValues<float[]> ravv, GraphIndex index, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        var usePq = ravv instanceof PQRandomAccessVectorValues;
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                NeighborQueue nn;
                if (usePq) {
                    NeighborSimilarity.ApproximateScoreFunction sf = (other) -> ((PQRandomAccessVectorValues) ravv).decodedSimilarity(other, queryVector, ds.similarityFunction);
                    nn = new GraphSearcher.Builder(index.getView())
                            .build()
                            .search(sf, efSearch, null, Integer.MAX_VALUE);
                } else {
                    nn = GraphSearcher.search(queryVector, efSearch, ravv, VectorEncoding.FLOAT32, ds.similarityFunction, index, null, Integer.MAX_VALUE);
                }

                int[] results;
                if (usePq && efSearch > topK) {
                    // re-rank the quantized results by the original similarity function
                    // Decorate
                    int[] raw = new int[nn.size()];
                    for (int j = raw.length - 1; j >= 0; j--) {
                        raw[j] = nn.pop();
                    }
                    // Pair each item in `raw` with its computed similarity
                    Map.Entry<Integer, Double>[] decorated = new AbstractMap.SimpleEntry[raw.length];
                    for (int j = 0; j < raw.length; j++) {
                        double similarity = ds.similarityFunction.compare(queryVector, ds.baseVectors.get(raw[j]));
                        decorated[j] = new AbstractMap.SimpleEntry<>(raw[j], similarity);
                    }
                    // Sort based on the computed similarity
                    Arrays.sort(decorated, (p1, p2) -> Double.compare(p2.getValue(), p1.getValue())); // Note the order for reversed sort
                    // Undecorate
                    results = Arrays.stream(decorated).mapToInt(Map.Entry::getKey).toArray();
                } else {
                    results = new int[nn.size()];
                    for (int j = results.length - 1; j >= 0; j--) {
                        results[j] = nn.pop();
                    }
                }

                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, results, gt);
                topKfound.add(n);
                nodesVisited.add(nn.visitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    record DataSet(VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) { }

    private static DataSet load(String pathStr) {
        // infer the similarity
        VectorSimilarityFunction similarityFunction;
        if (pathStr.contains("angular")) {
            similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        } else if (pathStr.contains("euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        } else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + pathStr);
        }

        // read the data
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        try (HdfFile hdf = new HdfFile(Paths.get(pathStr))) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        List<float[]> scrubbedBaseVectors;
        List<float[]> scrubbedQueryVectors;
        List<Set<Integer>> gtSet;
        if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            // verify that vectors are normalized and sane
            scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
            scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
            gtSet = new ArrayList<>(groundTruth.length);
            // remove zero vectors, noting that this will change the indexes of the ground truth answers
            Map<Integer, Integer> rawToScrubbed = new HashMap<>();
            {
                int j = 0;
                for (int i = 0; i < baseVectors.length; i++) {
                    float[] v = baseVectors[i];
                    if (Math.abs(normOf(v)) > 1e-5) {
                        scrubbedBaseVectors.add(v);
                        rawToScrubbed.put(i, j++);
                    }
                }
            }
            for (int i = 0; i < queryVectors.length; i++) {
                float[] v = queryVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedQueryVectors.add(v);
                    var gt = new HashSet<Integer>();
                    for (int j = 0; j < groundTruth[i].length; j++) {
                        gt.add(rawToScrubbed.get(groundTruth[i][j]));
                    }
                    gtSet.add(gt);
                }
            }
            // now that the zero vectors are removed, we can normalize
            if (Math.abs(normOf(baseVectors[0]) - 1.0) > 1e-5) {
                normalizeAll(scrubbedBaseVectors);
                normalizeAll(scrubbedQueryVectors);
            }
            assert scrubbedQueryVectors.size() == gtSet.size();
        } else {
            scrubbedBaseVectors = Arrays.asList(baseVectors);
            scrubbedQueryVectors = Arrays.asList(queryVectors);
            gtSet = new ArrayList<>(groundTruth.length);
            for (int[] gt : groundTruth) {
                var gtSetForQuery = new HashSet<Integer>();
                for (int i : gt) {
                    gtSetForQuery.add(i);
                }
                gtSet.add(gtSetForQuery);
            }
        }

        System.out.format("%n%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        return new DataSet(similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
        var files = List.of(
                "hdf5/nytimes-256-angular.hdf5",
                "hdf5/glove-100-angular.hdf5",
                "hdf5/glove-200-angular.hdf5",
                "hdf5/sift-128-euclidean.hdf5");
        var mGrid = List.of(8, 12, 16, 24, 32, 48, 64);
        var efConstructionGrid = List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchFactor = List.of(1, 2, 4);
        var pqOptions = List.of(false, true);
        // large files not yet supported
//                "hdf5/deep-image-96-angular.hdf5",
//                "hdf5/gist-960-euclidean.hdf5");
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, pqOptions, efSearchFactor);
        }

        // tiny dataset, don't waste time building a huge index
        files = List.of("hdf5/fashion-mnist-784-euclidean.hdf5");
        mGrid = List.of(8, 12, 16, 24);
        efConstructionGrid = List.of(40, 60, 80, 100, 120, 160);
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, pqOptions, efSearchFactor);
        }
    }

    private static void gridSearch(String f, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> pqOptions, List<Integer> efSearchFactor) throws ExecutionException, InterruptedException {
        var ds = load(f);

        var start = System.nanoTime();
        var pqDims = ds.baseVectors.get(0).length / 2;
        ProductQuantization pq = new ProductQuantization(ds.baseVectors, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(ds.baseVectors);
        System.out.format("PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var pqvv = new PQRandomAccessVectorValues(pq, quantizedVectors);

        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, pqOptions, efSearchFactor, ds, pqvv);
            }
        }
    }
}
