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

package com.github.jbellis.jvector.example;

import com.github.jbellis.jvector.disk.CachingGraphIndex;
import com.github.jbellis.jvector.disk.CompressedVectors;
import com.github.jbellis.jvector.disk.OnDiskGraphIndex;
import com.github.jbellis.jvector.example.util.MappedRandomAccessReader;
import com.github.jbellis.jvector.graph.*;
import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;
import com.github.jbellis.jvector.vector.VectorUtil;
import io.jhdf.HdfFile;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    private static void testRecall(int M, int efConstruction, List<Boolean> diskOptions, List<Integer> efSearchOptions, DataSet ds, CompressedVectors cv, Path testDirectory) throws IOException {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        var topK = ds.groundTruth.get(0).size();

        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();
        System.out.format("Build M=%d ef=%d in %.2fs with %.2f short edges%n",
                M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);

        var graphPath = testDirectory.resolve("graph" + M + efConstruction + ds.name);
        try {
            DataOutputStream outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)));
            OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
            outputStream.flush();
            var marr = new MappedRandomAccessReader(graphPath.toAbsolutePath().toString());
            var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(marr::duplicate, 0));

            int queryRuns = 10;
            for (int overquery : efSearchOptions) {
                for (boolean useDisk : diskOptions) {
                    start = System.nanoTime();
                    var pqr = performQueries(ds, floatVectors, useDisk ? cv : null, useDisk ? onDiskGraph : onHeapGraph, topK, topK * overquery, queryRuns);
                    var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
                    System.out.format("  Query PQ=%b top %d/%d recall %.4f in %.2fs after %s nodes visited%n",
                            useDisk, topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
                }
            }
        }
        finally {
            Files.deleteIfExists(graphPath);
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    static class ResultSummary {
        final int topKFound;
        final long nodesVisited;

        ResultSummary(int topKFound, long nodesVisited) {
            this.topKFound = topKFound;
            this.nodesVisited = nodesVisited;
        }
    }

    private static long topKCorrect(int topK, int[] resultNodes, Set<Integer> gt) {
        int count = Math.min(resultNodes.length, topK);
        // stream the first count results into a Set
        var resultSet = Arrays.stream(resultNodes, 0, count)
                .boxed()
                .collect(Collectors.toSet());
        assert resultSet.size() == count : String.format("%s duplicate results out of %s", count - resultSet.size(), count);
        return resultSet.stream().filter(gt::contains).count();
    }

    private static long topKCorrect(int topK, SearchResult.NodeScore[] nn, Set<Integer> gt) {
        var a = Arrays.stream(nn).mapToInt(nodeScore -> nodeScore.node).toArray();
        return topKCorrect(topK, a, gt);
    }

    private static ResultSummary performQueries(DataSet ds, RandomAccessVectorValues<float[]> exactVv, CompressedVectors cv, GraphIndex<float[]> index, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                SearchResult sr;
                if (cv != null) {
                    var view = index.getView();
                    NeighborSimilarity.ApproximateScoreFunction sf = (other) -> cv.decodedSimilarity(other, queryVector, ds.similarityFunction);
                    NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> ds.similarityFunction.compare(queryVector, vectors.get(j));
                    sr = new GraphSearcher.Builder(view)
                            .build()
                            .search(sf, rr, efSearch, null);
                } else {
                    sr = GraphSearcher.search(queryVector, efSearch, exactVv, VectorEncoding.FLOAT32, ds.similarityFunction, index, null);
                }

                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum()); // TODO do we care enough about visited count to hack it back into searcher?
    }

    static class DataSet {
        final String name;
        final VectorSimilarityFunction similarityFunction;
        final List<float[]> baseVectors;
        final List<float[]> queryVectors;
        final List<Set<Integer>> groundTruth;

        DataSet(String name, VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) {
            this.name = name;
            this.similarityFunction = similarityFunction;
            this.baseVectors = baseVectors;
            this.queryVectors = queryVectors;
            this.groundTruth = groundTruth;
        }

    }
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
        Path path = Paths.get(pathStr);
        try (HdfFile hdf = new HdfFile(path)) {
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

        return new DataSet(path.getFileName().toString(), similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());
        var files = List.of(
                "../hdf5/nytimes-256-angular.hdf5",
                "../hdf5/glove-100-angular.hdf5",
                "../hdf5/glove-200-angular.hdf5",
                "../hdf5/sift-128-euclidean.hdf5");
        var mGrid = List.of(8, 12, 16, 24, 32, 48, 64);
        var efConstructionGrid = List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchFactor = List.of(1, 2);
        var diskOptions = List.of(false, true);
        // large files not yet supported
//                "hdf5/deep-image-96-angular.hdf5",
//                "hdf5/gist-960-euclidean.hdf5");
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, diskOptions, efSearchFactor);
        }

        // tiny dataset, don't waste time building a huge index
        files = List.of("../hdf5/fashion-mnist-784-euclidean.hdf5");
        mGrid = List.of(8, 12, 16, 24);
        efConstructionGrid = List.of(40, 60, 80, 100, 120, 160);
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, diskOptions, efSearchFactor);
        }
    }

    private static void gridSearch(String f, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        var ds = load(f);

        var start = System.nanoTime();
        var pqDims = ds.baseVectors.get(0).length / 2;
        ProductQuantization pq = new ProductQuantization(ds.baseVectors, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(ds.baseVectors);
        System.out.format("PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var compressedVectors = new CompressedVectors(pq, quantizedVectors);

        var testDirectory = Files.createTempDirectory("BenchGraphDir");

        try {
            for (int M : mGrid) {
                for (int beamWidth : efConstructionGrid) {
                    testRecall(M, beamWidth, diskOptions, efSearchFactor, ds, compressedVectors, testDirectory);
                }
            }
        } finally {
            Files.delete(testDirectory);
        }
    }
}
