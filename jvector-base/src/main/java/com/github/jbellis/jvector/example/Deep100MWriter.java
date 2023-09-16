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
import com.github.jbellis.jvector.example.util.DataSet;
import com.github.jbellis.jvector.example.util.Deep1BLoader;
import com.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import com.github.jbellis.jvector.graph.GraphIndex;
import com.github.jbellis.jvector.graph.GraphSearcher;
import com.github.jbellis.jvector.graph.NeighborSimilarity;
import com.github.jbellis.jvector.graph.SearchResult;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.File;
import java.io.IOException;
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
public class Deep100MWriter {
    private static void testRecall(int M, int efConstruction, List<Boolean> diskOptions, List<Integer> efSearchOptions, DataSet ds, CompressedVectors cv, Path testDirectory) throws IOException {
        var topK = ds.groundTruth.get(0).size();

        var graphPath = testDirectory.resolve("graph" + M + efConstruction + ds.name);
        var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0));

        int queryRuns = 1;
        for (int overquery : efSearchOptions) {
            for (boolean useDisk : diskOptions) {
                var start = System.nanoTime();
                var pqr = performQueries(ds, cv, onDiskGraph, topK, topK * overquery, queryRuns);
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

    private static ResultSummary performQueries(DataSet ds, CompressedVectors cv, GraphIndex<float[]> index, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                SearchResult sr;
                var view = index.getView();
                NeighborSimilarity.ApproximateScoreFunction sf = (other) -> cv.decodedSimilarity(other, queryVector, ds.similarityFunction);
                NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> ds.similarityFunction.compare(queryVector, vectors.get(j));
                sr = new GraphSearcher.Builder(view)
                        .build()
                        .search(sf, rr, efSearch, null);
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

        var queryVectors = Deep1BLoader.readFBin("bigann-data/deep1b/query.public.10K.fbin", 10_000);
        var gt = Deep1BLoader.readGT("bigann-data/deep1b/deep-100M");

        var ds = new DataSet("Deep100M", VectorSimilarityFunction.EUCLIDEAN, null, queryVectors, gt);
        gridSearch(ds, mGrid, efConstructionGrid, diskOptions, efSearchFactor);
    }

    private static void gridSearch(DataSet ds, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        var testDirectory = new File("jvectorindex").toPath();
        CompressedVectors compressedVectors;
        try (var rsf = ReaderSupplierFactory.open(testDirectory.resolve("compressedVectors"))) {
            compressedVectors = CompressedVectors.load(rsf.get(), 0);
            System.out.println("Compressed vectors loaded");
        }

        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, diskOptions, efSearchFactor, ds, compressedVectors, testDirectory);
            }
        }
    }
}
