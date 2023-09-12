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
import io.github.jbellis.jvector.disk.CompressedVectors;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
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
    private static int queryRuns = 10;

    private static void testRecall(GraphResult graph, int topK, int M, int efConstruction, boolean useDisk, int overquery, DataSet ds, CompressedVectors cv) throws IOException {
        long start = System.nanoTime();
        var pqr = performQueries(ds, graph.floatVectors, useDisk ? cv : null, useDisk ? graph.onDiskGraph : graph.onHeapGraph, topK, topK * overquery, queryRuns);
        var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
        System.out.format("  Query PQ=%b top %d/%d recall %.4f in %.2fs after %s nodes visited%n",
                useDisk, topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
    }

    private static GraphResult buildGraph(int M, int efConstruction, DataSet ds, Path graphPath) throws IOException {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();

        String details = String.format("Build M=%d ef=%d in %.2fs with %.2f short edges",
                M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);
        DataOutputStream outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)));
        OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
        outputStream.flush();
        var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0));
        GraphResult result = new GraphResult(floatVectors, onHeapGraph, onDiskGraph, details);
        return result;
    }

    private static class GraphResult {
        public final ListRandomAccessVectorValues floatVectors;
        public final OnHeapGraphIndex<float[]> onHeapGraph;
        public final CachingGraphIndex onDiskGraph;

        public final String details;

        public GraphResult(ListRandomAccessVectorValues floatVectors, OnHeapGraphIndex<float[]> onHeapGraph,
                           CachingGraphIndex onDiskGraph, String details) {
            this.floatVectors = floatVectors;
            this.onHeapGraph = onHeapGraph;
            this.onDiskGraph = onDiskGraph;
            this.details = details;
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

    public static void main(String[] args) throws IOException {
        var files = args == null || args.length == 0
                ? List.of(
                    "../ivec2/100k/pages_ada_002_100k",
                    "../hdf5/nytimes-256-angular.hdf5",
                    "../hdf5/glove-100-angular.hdf5",
                    "../hdf5/glove-200-angular.hdf5",
                    "../hdf5/fashion-mnist-784-euclidean.hdf5",
                    "../hdf5/sift-128-euclidean.hdf5"
                    )
                : Arrays.asList(args);

        var mGrid = List.of(16, 24, 32, 48);
        var efConstructionGrid = List.of(100, 200, 400);
        var efSearchFactor = List.of(1, 2);
        var diskOptions = List.of(true, false);
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, diskOptions, efSearchFactor);
        }

    }

    private static DataSet load(String pathStr) throws IOException {
        boolean isHdf5 = pathStr.endsWith("hdf5");
        if (isHdf5) {
            return Hdf5Loader.load(pathStr);
        } else {
            // assuming sift
            final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;

            float[][] baseVectors = SiftLoader.readFvecsAsArray(String.format("%s_base_vectors.fvec", pathStr));
            float[][] queryVectors = SiftLoader.readFvecsAsArray(String.format("%s_query_vectors_10k.fvec", pathStr));
            int[][] groundTruth = SiftLoader.readIvecsAsArray(String.format("%s_indices_query_vectors_10k.ivec", pathStr));

            return DataSet.getScrubbedDataSet(pathStr, similarityFunction, baseVectors, queryVectors, groundTruth);
        }

    }

    private static void gridSearch(String f, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        var ds = load(f);

        int originalDimension = ds.baseVectors.get(0).length;
        List<Integer> pqDimensions = new ArrayList<>();
        int dims = ds.baseVectors.get(0).length;
        for (int i = 2; i <= 32; i *= 2) {
            if (dims / i > 1) {
                pqDimensions.add(dims / i);
            }
        }
        
        var testDirectory = Files.createTempDirectory("BenchGraphDir");

        try {
            for (int M : mGrid) {
                for (int beamWidth : efConstructionGrid) {
                    var graphPath = testDirectory.resolve("graph" + M + beamWidth + ds.name);
                    var topK = ds.groundTruth.get(0).size();
                    GraphResult graph = buildGraph(M, beamWidth, ds, graphPath);

                    for (boolean diskOpt : diskOptions) {
                        try {
                            if (diskOpt) {
                                for (var pqDims : pqDimensions) {
                                    var start = System.nanoTime();
                                    ListRandomAccessVectorValues ravv = new ListRandomAccessVectorValues(ds.baseVectors, originalDimension);
                                    ProductQuantization pq = ProductQuantization.compute(ravv, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
                                    String pgBuildDetails = String.format("PQ@%s build %.2fs", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

                                    start = System.nanoTime();
                                    var quantizedVectors = pq.encodeAll(ds.baseVectors);
                                    String pgEncodeDetails = String.format("PQ encode %.2fs", (System.nanoTime() - start) / 1_000_000_000.0);

                                    var compressedVectors = new CompressedVectors(pq, quantizedVectors);

                                    for (var overquery: efSearchFactor) {
                                        System.out.println(ds.details);
                                        System.out.println(graph.details);
                                        System.out.println(pgBuildDetails);
                                        System.out.println(pgEncodeDetails);
                                        testRecall(graph, topK, M, beamWidth, true, overquery, ds, compressedVectors);
                                        System.out.println();
                                    }
                                }
                            } else {
                                for (var overquery: efSearchFactor) {
                                    System.out.println(ds.details);
                                    System.out.println(graph.details);
                                    testRecall(graph, topK, M, beamWidth, false, overquery, ds, null);
                                    System.out.println();
                                }
                            }
                        } catch (Throwable t) {
                                System.out.println("Exception " + t);
                                t.printStackTrace();
                        } finally {
                            Files.deleteIfExists(graphPath);
                        }
                    }
                }
            }
        } catch (Throwable t) {
            System.out.println("Exception " + t);
            t.printStackTrace();
        } finally {
            Files.deleteIfExists(testDirectory);
        }
    }
}
