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
import io.github.jbellis.jvector.example.util.Deep1BLoader;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NeighborSimilarity;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Deep100MWriter {
    private static void testRecall(int M, int efConstruction, List<Boolean> diskOptions, List<Integer> efSearchOptions, DataSet ds, CompressedVectors cv, Path testDirectory) throws IOException {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        var topK = ds.groundTruth.get(0).size();

        var startBuild = System.nanoTime();
        AtomicInteger completed = new AtomicInteger();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.2f, 1.4f);
        IntStream.range(0, ds.baseVectors.size()).parallel().forEach(i -> {
            builder.addGraphNode(i, floatVectors);

            var n = completed.incrementAndGet();
            if (n % 10_000 == 0) {
                long elapsedTime = System.nanoTime() - startBuild;
                long estimatedRemainingTime = (long)((ds.baseVectors.size() - n) * ((double)elapsedTime / n));
                // Convert time from nanoseconds to seconds
                double elapsedTimeInSeconds = elapsedTime / 1_000_000_000.0;
                double estimatedRemainingTimeInSeconds = estimatedRemainingTime / 1_000_000_000.0;
                System.out.printf("%,d completed in %.2f seconds. Estimated remaining time: %.2f seconds%n", n, elapsedTimeInSeconds, estimatedRemainingTimeInSeconds);
            }
        });
        builder.complete();
        var onHeapGraph = builder.getGraph();

        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();
        System.out.format("Build M=%d ef=%d in %.2fs with %.2f short edges%n",
                M, efConstruction, (System.nanoTime() - startBuild) / 1_000_000_000.0, avgShortEdges);

        var graphPath = testDirectory.resolve("graph" + M + efConstruction + ds.name);
        if (!Files.exists(graphPath.getParent())) {
            Files.createDirectories(graphPath.getParent());
        }
        DataOutputStream outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)));
        OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
        outputStream.flush();
        var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0));

        int queryRuns = 2;
        for (int overquery : efSearchOptions) {
            for (boolean useDisk : diskOptions) {
                var start = System.nanoTime();
                var pqr = performQueries(ds, floatVectors, useDisk ? cv : null, useDisk ? onDiskGraph : onHeapGraph, topK, topK * overquery, queryRuns);
                var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
                System.out.format("  Query PQ=%b top %d/%d recall %.4f in %.2fs after %s nodes visited%n",
                        useDisk, topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
            }
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
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32);
        var efConstructionGrid = List.of(100);
        var efSearchFactor = List.of(1);
        var diskOptions = List.of(true);

        var baseVectors = Deep1BLoader.readFBin("bigann-data/deep1b/base.1B.fbin.crop_nb_10000000", 10_000_000);
        var queryVectors = Deep1BLoader.readFBin("bigann-data/deep1b/query.public.10K.fbin", 10_000);
        var gt = Deep1BLoader.readGT("bigann-data/deep1b/deep-10M");

        var ds = new DataSet("Deep100M", VectorSimilarityFunction.EUCLIDEAN, baseVectors, queryVectors, gt);
        gridSearch(ds, mGrid, efConstructionGrid, diskOptions, efSearchFactor);
    }

    private static void gridSearch(DataSet ds, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        var start = System.nanoTime();
        var pqDims = ds.baseVectors.get(0).length / 4;
        var ravv = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        ProductQuantization pq = ProductQuantization.compute(ravv, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(ds.baseVectors);
        System.out.format("PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var compressedVectors = new CompressedVectors(pq, quantizedVectors);
        try (var out = new BufferedOutputStream(Files.newOutputStream(new File("jvectorindex/compressedVectors").toPath()))) {
            compressedVectors.write(new DataOutputStream(out));
        }

        var testDirectory = new File("jvectorindex").toPath();

        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, diskOptions, efSearchFactor, ds, compressedVectors, testDirectory);
            }
        }
    }
}
