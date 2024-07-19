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

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.pq.VectorCompressor;
import io.github.jbellis.jvector.spann.IVFIndex;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;

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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests a grid of configurations against a dataset
 */
public class Grid {
    private static final VectorUtilSupport vectorUtilSupport = VectorizationProvider.getInstance().getVectorUtilSupport();

    private static final String pqCacheDir = "pq_cache";

    private static final String dirPrefix = "BenchGraphDir";

    private static final int searchCentroids = 8; // TODO query pruning

    static void runAll(DataSet ds,
                       List<Integer> mGrid,
                       List<Integer> efConstructionGrid,
                       List<? extends Set<FeatureId>> featureSets,
                       List<Function<DataSet, CompressorParameters>> buildCompressors,
                       List<Function<DataSet, CompressorParameters>> compressionGrid,
                       List<Double> efSearchFactor) throws IOException
    {
        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            runOne(ds, testDirectory);
        } finally {
            Files.delete(testDirectory);
            cachedCompressors.clear();
        }
    }

    static void runOne(DataSet ds, Path testDirectory) throws IOException
    {
        var ivf = IVFIndex.build(ds.getBaseRavv(), ds.similarityFunction);
        long start = System.nanoTime();
        var topK = ds.groundTruth.get(0).size();
        var pqr = performQueries(ivf, ds, topK, 2);
        var recall = ((double) pqr.topKFound) / (2 * ds.queryVectors.size() * topK);
        System.out.format(" Query top %d recall %.4f in %.2fs after %,d nodes visited%n",
                          topK, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

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

    private static ResultSummary performQueries(IVFIndex ivf, DataSet ds, int topK, int queryRuns) {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                SearchResult sr;
                sr = ivf.search(queryVector, topK, searchCentroids);

                // process search result
                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum());
    }

    static class ResultSummary {
        final int topKFound;
        final long nodesVisited;

        ResultSummary(int topKFound, long nodesVisited) {
            this.topKFound = topKFound;
            this.nodesVisited = nodesVisited;
        }
    }
}
