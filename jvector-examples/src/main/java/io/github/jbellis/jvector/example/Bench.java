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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
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
            try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
                OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
            }
            try (var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0))) {
                int queryRuns = 2;
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
        }
        finally {
            Files.deleteIfExists(graphPath);
        }
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
                    NeighborSimilarity.ApproximateScoreFunction sf = cv.approximateScoreFunctionFor(queryVector, ds.similarityFunction);
                    NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> ds.similarityFunction.compare(queryVector, vectors.get(j));
                    sr = new GraphSearcher.Builder<>(view)
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

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(8, 12, 16, 24, 32, 48, 64);
        var efConstructionGrid = List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchGrid = List.of(1, 2);
        var diskGrid = List.of(false, true);

        // this dataset contains more than 10k query vectors, so we limit it with .subList
        var adaSet = loadWikipediaData();
        gridSearch(adaSet, mGrid, efConstructionGrid, diskGrid, efSearchGrid);

        var files = List.of(
                // large files not yet supported
                // "hdf5/deep-image-96-angular.hdf5",
                // "hdf5/gist-960-euclidean.hdf5",
                "hdf5/nytimes-256-angular.hdf5",
                "hdf5/glove-100-angular.hdf5",
                "hdf5/glove-200-angular.hdf5",
                "hdf5/sift-128-euclidean.hdf5");
        for (var f : files) {
            gridSearch(Hdf5Loader.load(f), mGrid, efConstructionGrid, diskGrid, efSearchGrid);
        }

        // tiny dataset, don't waste time building a huge index
        files = List.of("hdf5/fashion-mnist-784-euclidean.hdf5");
        mGrid = List.of(8, 12, 16, 24);
        efConstructionGrid = List.of(40, 60, 80, 100, 120, 160);
        for (var f : files) {
            gridSearch(Hdf5Loader.load(f), mGrid, efConstructionGrid, diskGrid, efSearchGrid);
        }
    }

    private static DataSet loadWikipediaData() throws IOException {
        var baseVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_query_vectors_10k.fvec").subList(0, 10_000);
        var gt = SiftLoader.readIvecs("fvec/pages_ada_002_100k_indices_query_vectors_10k.ivec").subList(0, 10_000);
        var ds = new DataSet("wikipedia",
                             VectorSimilarityFunction.DOT_PRODUCT,
                             baseVectors,
                             queryVectors,
                             gt);
        System.out.format("%nWikipedia: %d base and %d query vectors loaded, dimensions %d%n",
                          baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);
        return ds;
    }

    private static void gridSearch(DataSet ds, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        var start = System.nanoTime();
        int originalDimension = ds.baseVectors.get(0).length;
        var pqDims = originalDimension / 4;
        ListRandomAccessVectorValues ravv = new ListRandomAccessVectorValues(ds.baseVectors, originalDimension);
        ProductQuantization pq = ProductQuantization.compute(ravv, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(ds.baseVectors);
        var compressedVectors = new CompressedVectors(pq, quantizedVectors);
        System.out.format("PQ encoded %d vectors [%.2f MB] in %.2fs,%n", ds.baseVectors.size(), (compressedVectors.memorySize()/1024f/1024f) , (System.nanoTime() - start) / 1_000_000_000.0);

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
