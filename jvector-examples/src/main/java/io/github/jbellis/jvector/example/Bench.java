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

import io.github.jbellis.jvector.disk.CachingADCGraphIndex;
import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskADCGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetCreator;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.Reranker;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.pq.VectorCompressor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static void testRecall(int M,
                                   int efConstruction,
                                   List<Function<DataSet, VectorCompressor<?>>> compressionGrid,
                                   List<Integer> efSearchOptions,
                                   DataSet ds,
                                   Path testDirectory) throws IOException
    {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length());
        var topK = ds.groundTruth.get(0).size();

        var graphPath = testDirectory.resolve("graph" + M + efConstruction + ds.name);
        var fusedGraphPath = testDirectory.resolve("fusedgraph" + M + efConstruction + ds.name);
        try {
            for (var cf : compressionGrid) {
                // compress vectors
                var compressor = getCompressor(cf, ds);
                CompressedVectors cv;
                var start = System.nanoTime();
                var quantizedVectors = compressor.encodeAll(ds.baseVectors);
                cv = compressor.createCompressedVectors(quantizedVectors);
                System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, ds.baseVectors.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);

                // build
                start = System.nanoTime();
                var builder = new GraphIndexBuilder(floatVectors, ds.similarityFunction, M, efConstruction, 1.2f, 1.2f);
                var onHeapGraph = builder.build(ds.getBaseRavv());
                System.out.format("Build M=%d ef=%d in %.2fs with avg degree %.2f and %.2f short edges%n",
                                  M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, onHeapGraph.getAverageDegree(), onHeapGraph.getAverageShortEdges());

                // write
                try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
                    OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
                }

                // query
                try (var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex(ReaderSupplierFactory.open(graphPath), 0))) {
                    int queryRuns = 2;
                    for (int overquery : efSearchOptions) {
                        if (compressor == null) {
                            // include both in-memory and on-disk search of uncompressed vectors
                            start = System.nanoTime();
                            var pqr = performQueries(ds, floatVectors, cv, onHeapGraph, topK, topK * overquery, queryRuns);
                            var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
                            System.out.format("  Query %s top %d/%d recall %.4f in %.2fs after %,d nodes visited%n",
                                              "(memory)", topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
                        }
                        start = System.nanoTime();
                        var pqr = performQueries(ds, floatVectors, cv, onDiskGraph, topK, topK * overquery, queryRuns);
                        var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
                        System.out.format("  Query %stop %d/%d recall %.4f in %.2fs after %,d nodes visited%n",
                                          compressor == null ? "(disk) " : "", topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(graphPath);
            Files.deleteIfExists(fusedGraphPath);
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    private static final Map<Function<DataSet, VectorCompressor<?>>, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();
    private static VectorCompressor<?> getCompressor(Function<DataSet, VectorCompressor<?>> cf, DataSet ds) {
        if (cf == null) {
            return null;
        }
        return cachedCompressors.computeIfAbsent(cf, __ -> {
            var start = System.nanoTime();
            var compressor = cf.apply(ds);
            System.out.format("%s build in %.2fs,%n", compressor, (System.nanoTime() - start) / 1_000_000_000.0);
            return compressor;
        });
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

    private static ResultSummary performQueries(DataSet ds, RandomAccessVectorValues exactVv, CompressedVectors cv, GraphIndex index, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                SearchResult sr;
                if (cv != null) {
                    try (var view = index.getView()) {
                        ScoreFunction.ApproximateScoreFunction sf;
                        if (index instanceof CachingADCGraphIndex) {
                            sf = ((CachingADCGraphIndex.CachedView) view).approximateScoreFunctionFor(queryVector, ds.similarityFunction);
                        } else {
                            sf = cv.approximateScoreFunctionFor(queryVector, ds.similarityFunction);
                        }
                        var rr = Reranker.from(queryVector, ds.similarityFunction, view);
                        var ssp = new SearchScoreProvider(sf, rr);
                        sr = new GraphSearcher.Builder(view)
                                .build()
                                .search(ssp, efSearch, Bits.ALL);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    sr = GraphSearcher.search(queryVector, efSearch, exactVv, ds.similarityFunction, index, Bits.ALL);
                }

                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum());
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
        var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchGrid = List.of(1, 2);
        List<Function<DataSet, VectorCompressor<?>>> compressionGrid = Arrays.asList(
                /*ds -> ProductQuantization.compute(ds.getBaseRavv(), ds.getDimension() / 4, 32,
                                                  ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN),*/
                ds -> ProductQuantization.compute(ds.getBaseRavv(), ds.getDimension() / 8,
                                   ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN));

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        // large embeddings calculated by Neighborhood Watch.  100k files by default; 1M also available
        var nwFiles = List.of(
//                "colbert-10M", // WIP
                "openai-v3-large-3072-100k",
                "openai-v3-large-1536-100k",
                "openai-v3-small-100k",
                "e5-small-v2-100k",
                "e5-base-v2-100k",
                "e5-large-v2-100k",
                "gecko-100k",
                "ada002-100k");
        for (var nwDatasetName : nwFiles) {
            if (pattern.matcher(nwDatasetName).find()) {
                var mfd = DownloadHelper.maybeDownloadFvecs(nwDatasetName);
                gridSearch(mfd.load(), compressionGrid, mGrid, efConstructionGrid, efSearchGrid);
                cachedCompressors.clear();
            }
        }

        // smaller vectors from ann-benchmarks
        var hdf5Files = List.of(
                // large files not yet supported
                // "hdf5/deep-image-96-angular.hdf5",
                // "hdf5/gist-960-euclidean.hdf5",
                "glove-25-angular.hdf5",
                "glove-50-angular.hdf5",
                "lastfm-64-dot.hdf5",
                "glove-100-angular.hdf5",
                "glove-200-angular.hdf5",
                "nytimes-256-angular.hdf5",
                "sift-128-euclidean.hdf5");
        for (var f : hdf5Files) {
            if (pattern.matcher(f).find()) {
                DownloadHelper.maybeDownloadHdf5(f);
                gridSearch(Hdf5Loader.load(f), compressionGrid, mGrid, efConstructionGrid, efSearchGrid);
                cachedCompressors.clear();
            }
        }

        // 2D grid, built and calculated at runtime
        if (pattern.matcher("2dgrid").find()) {
            compressionGrid = Arrays.asList(null,
                                            ds -> ProductQuantization.compute(ds.getBaseRavv(), ds.getDimension(), true));
            var grid2d = DataSetCreator.create2DGrid(4_000_000, 10_000, 100);
            gridSearch(grid2d, compressionGrid, mGrid, efConstructionGrid, efSearchGrid);
            cachedCompressors.clear();
        }
    }

    private static void gridSearch(DataSet ds,
                                   List<Function<DataSet, VectorCompressor<?>>> compressionGrid,
                                   List<Integer> mGrid,
                                   List<Integer> efConstructionGrid,
                                   List<Integer> efSearchFactor) throws IOException
    {
        var testDirectory = Files.createTempDirectory("BenchGraphDir");
        try {
            for (int M : mGrid) {
                for (int efC : efConstructionGrid) {
                    testRecall(M, efC, compressionGrid, efSearchFactor, ds, testDirectory);
                }
            }
        } finally {
            Files.delete(testDirectory);
        }
    }
}
