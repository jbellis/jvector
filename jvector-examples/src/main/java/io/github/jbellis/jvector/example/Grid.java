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
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.pq.VectorCompressor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
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

    private static final double centroidFraction = 0.16; // TODO tune this for GPU
    private static final int nCentroidsPerVector = 4; // TODO replace with closure assignment
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
        var ivf = IVFIndex.build(ds);
        long start = System.nanoTime();
        var topK = ds.groundTruth.get(0).size();
        var pqr = performQueries(ivf, ds, topK, 2);
        var recall = ((double) pqr.topKFound) / (2 * ds.queryVectors.size() * topK);
        System.out.format(" Query top %d recall %.4f in %.2fs after %,d nodes visited%n",
                          topK, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
    }

    static class IVFIndex {
        private final GraphIndex index;
        private final RandomAccessVectorValues ravv;
        private final Int2ObjectHashMap<int[]> postings;
        private final ExplicitThreadLocal<GraphSearcher> searchers;
        private final VectorSimilarityFunction vsf;

        public IVFIndex(GraphIndex index, RandomAccessVectorValues ravv, Int2ObjectHashMap<int[]> postings, VectorSimilarityFunction vsf) {
            this.index = index;
            this.ravv = ravv;
            this.postings = postings;
            searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));
            this.vsf = vsf;
        }

        public static IVFIndex build(DataSet ds) {
            // build the graph index
            long start = System.nanoTime();
            var centroids = selectCentroids(ds.getBaseRavv(), centroidFraction);
            OnHeapGraphIndex index;
            try (var builder = new GraphIndexBuilder(ds.getBaseRavv(), ds.similarityFunction, 32, 100, 1.2f, 1.2f)) {
                index = builder.build(centroids);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            System.out.printf("Graph index built in %ss%n", (System.nanoTime() - start) / 1_000_000_000.0);

            // assign vectors to centroids
            start = System.nanoTime();
            var postings = new ConcurrentHashMap<Integer, Set<Integer>>();

            ThreadLocal<GraphSearcher> searchers = ThreadLocal.withInitial(() -> new GraphSearcher(index));
            IntStream.range(0, ds.getBaseRavv().size()).parallel().forEach(i -> {
                var v = ds.getBaseRavv().getVector(i);
                var ssp = SearchScoreProvider.exact(v, ds.similarityFunction, ds.getBaseRavv());
                var sr = searchers.get().search(ssp, nCentroidsPerVector, Bits.ALL);
                for (var ns : sr.getNodes()) {
                    postings.computeIfAbsent(ns.node, __ -> ConcurrentHashMap.newKeySet()).add(i);
                }
            });
            var optimizedPostings = new Int2ObjectHashMap<int[]>();
            postings.forEach((k, v) -> optimizedPostings.put(k, v.stream().mapToInt(Integer::intValue).toArray()));
            System.out.printf("Assigned vectors to centroids in %ss%n", (System.nanoTime() - start) / 1_000_000_000.0);

            return new IVFIndex(index, ds.getBaseRavv(), optimizedPostings, ds.similarityFunction);
        }

        public SearchResult search(VectorFloat<?> queryVector, int topK, int nCentroids) {
            var searcher = searchers.get();
            var ssp = SearchScoreProvider.exact(queryVector, vsf, ravv);
            Set<Integer> allPostings = ConcurrentHashMap.newKeySet();
            SearchResult centroidsResult = null;
            // search until we find a non-empty centroid
            while (true)
            {
                if (centroidsResult == null) {
                    centroidsResult = searcher.search(ssp, nCentroids, Bits.ALL);
                } else {
                    centroidsResult = searcher.resume(nCentroids, nCentroids);
                }
                // combine results from all centroids
                for (var ns : centroidsResult.getNodes()) {
                    var subPostings = postings.get(ns.node);
                    if (subPostings == null) {
                        continue;
                    }
                    allPostings.addAll(Arrays.stream(subPostings).boxed().collect(Collectors.toSet()));
                }
                if (!allPostings.isEmpty()) {
                    break;
                }
            }
            // sort postings by score
            var scoredPostings = allPostings.parallelStream()
                    .map(i -> new SearchResult.NodeScore(i, vsf.compare(queryVector, ravv.getVector(i))))
                    .sorted((a, b) -> Float.compare(b.score, a.score))
                    .limit(topK)
                    .toArray(SearchResult.NodeScore[]::new);
            Arrays.sort(scoredPostings, (a, b) -> Float.compare(b.score, a.score));
            return new SearchResult(Arrays.stream(scoredPostings).limit(topK).toArray(SearchResult.NodeScore[]::new),
                                    centroidsResult.getVisitedCount() + allPostings.size(),
                                    allPostings.size(),
                                    Float.POSITIVE_INFINITY);
        }
    }

    /**
     * @return the centroids to which we will assign the rest of the vectors
     */
    private static RandomAccessVectorValues selectCentroids(RandomAccessVectorValues baseRavv, double centroidFraction) {
        // this creates redundant indexes (since we are using source vectors as centroids, each will also
        // end up assigned to itself in the posting lists)
        // we will fix this by switching to HBC centroid selection
        //
        // in the meantime: this is worse than i thought, only about 20% of the centroids get all of the vectors mapped to them
        int nCentroids = (int) (baseRavv.size() * centroidFraction);
        Set<VectorFloat<?>> selected = ConcurrentHashMap.newKeySet();
        var R = new Random();
        while (selected.size() < nCentroids) {
            selected.add(baseRavv.getVector(R.nextInt(baseRavv.size())));
        }
        List<VectorFloat<?>> L = new ArrayList<>(selected);
        return new ListRandomAccessVectorValues(L, baseRavv.dimension());
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
