package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.CachingADCGraphIndex;
import io.github.jbellis.jvector.disk.CachingGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskADCGraphIndex;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.pq.VectorCompressor;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.types.VectorFloat;

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
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests a grid of configurations against a dataset
 */
public class Grid {
    private static final String pqCacheDir = "pq_cache";

    private static final String dirPrefix = "BenchGraphDir";

    static void runAll(DataSet ds,
                       List<Function<DataSet, CompressorParameters>> compressionGrid,
                       List<Integer> mGrid,
                       List<Integer> efConstructionGrid,
                       List<Integer> efSearchFactor) throws IOException
    {
        var testDirectory = Files.createTempDirectory(dirPrefix);
        try {
            for (int M : mGrid) {
                for (int efC : efConstructionGrid) {
                    runOneGraph(M, efC, compressionGrid, efSearchFactor, ds, testDirectory);
                }
            }
        } finally {
            Files.delete(testDirectory);
            cachedCompressors.clear();
        }
    }

    static void runOneGraph(int M,
                            int efConstruction,
                            List<Function<DataSet, CompressorParameters>> compressionGrid,
                            List<Integer> efSearchOptions,
                            DataSet ds,
                            Path testDirectory) throws IOException
    {
        var floatVectors = ds.getBaseRavv();
        var builder = new GraphIndexBuilder(floatVectors, ds.similarityFunction, M, efConstruction, 1.2f, 1.2f);
        var start = System.nanoTime();
        var onHeapGraph = builder.build(floatVectors);
        System.out.format("Build M=%d ef=%d in %.2fs with avg degree %.2f and %.2f short edges%n",
                          M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, onHeapGraph.getAverageDegree(), onHeapGraph.getAverageShortEdges());

        var graphPath = testDirectory.resolve("graph" + M + efConstruction + ds.name);
        var fusedGraphPath = testDirectory.resolve("fusedgraph" + M + efConstruction + ds.name);
        try {
            try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(graphPath)))) {
                OnDiskGraphIndex.write(onHeapGraph, floatVectors, outputStream);
            }

            for (var cpSupplier : compressionGrid) {
                var compressor = getCompressor(cpSupplier, ds);
                CompressedVectors cv = null;
                var fusedCompatible = compressor instanceof ProductQuantization && ((ProductQuantization) compressor).getClusterCount() == 32;
                if (compressor == null) {
                    System.out.format("Uncompressed vectors%n");
                } else {
                    start = System.nanoTime();
                    var quantizedVectors = compressor.encodeAll(ds.baseVectors);
                    cv = compressor.createCompressedVectors(quantizedVectors);
                    System.out.format("%s encoded %d vectors [%.2f MB] in %.2fs%n", compressor, ds.baseVectors.size(), (cv.ramBytesUsed() / 1024f / 1024f), (System.nanoTime() - start) / 1_000_000_000.0);

                    if (fusedCompatible) {
                        try (var outputStream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(fusedGraphPath)))) {
                            OnDiskADCGraphIndex.write(onHeapGraph, floatVectors, (PQVectors) cv, outputStream);
                        }
                    }
                }

                try (var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex(ReaderSupplierFactory.open(graphPath), 0));
                     var onDiskFusedGraph = fusedCompatible ? new CachingADCGraphIndex(new OnDiskADCGraphIndex(ReaderSupplierFactory.open(fusedGraphPath), 0)) : null) {
                    List<GraphIndex> graphs = new ArrayList<>();
                    graphs.add(onDiskGraph);
                    if (onDiskFusedGraph != null) {
                        graphs.add(onDiskFusedGraph);
                    }
                    if (cv == null) {
                        graphs.add(onHeapGraph); // if we have no cv, compare on-heap/on-disk with exact searches
                    }
                    for (var g : graphs) {
                        var cs = new ConfiguredSystem(ds, g, cv);
                        testConfiguration(cs, efSearchOptions);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(graphPath);
            Files.deleteIfExists(fusedGraphPath);
        }
    }

    // avoid recomputing the compressor repeatedly (this is a relatively small memory footprint)
    static final Map<String, VectorCompressor<?>> cachedCompressors = new IdentityHashMap<>();

    private static void testConfiguration(ConfiguredSystem cs, List<Integer> efSearchOptions) {
        var topK = cs.ds.groundTruth.get(0).size();
        System.out.format("Using %s:%n", cs.index);
        for (int overquery : efSearchOptions) {
            var start = System.nanoTime();
            var pqr = performQueries(cs, topK, topK * overquery, 2);
            var recall = ((double) pqr.topKFound) / (2 * cs.ds.queryVectors.size() * topK);
            System.out.format(" Query top %d/%d recall %.4f in %.2fs after %,d nodes visited%n",
                              topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);

        }
    }

    private static VectorCompressor<?> getCompressor(Function<DataSet, CompressorParameters> cpSupplier, DataSet ds) {
        var cp = cpSupplier.apply(ds);
        if (!cp.supportsCaching()) {
            return cp.computeCompressor(ds);
        }

        var fname = cp.idStringFor(ds);
        return cachedCompressors.computeIfAbsent(fname, __ -> {
            var path = Paths.get(pqCacheDir).resolve(fname);
            if (path.toFile().exists()) {
                try {
                    try (var readerSupplier = ReaderSupplierFactory.open(path)) {
                        try (var rar = readerSupplier.get()) {
                            var pq = ProductQuantization.load(rar);
                            System.out.format("%s loaded from %s%n", pq, fname);
                            return pq;
                        }
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            var start = System.nanoTime();
            var compressor = cp.computeCompressor(ds);
            System.out.format("%s build in %.2fs,%n", compressor, (System.nanoTime() - start) / 1_000_000_000.0);
            if (cp.supportsCaching()) {
                try {
                    Files.createDirectories(path.getParent());
                    try (var writer = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
                        compressor.write(writer);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            return compressor;
        });
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

    private static ResultSummary performQueries(ConfiguredSystem cs, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, cs.ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = cs.ds.queryVectors.get(i);
                SearchResult sr;
                if (cs.cv != null) {
                    try (var view = cs.index.getView()) {
                        var sf = cs.scoreProviderFor(queryVector, view);
                        var searcher = new GraphSearcher.Builder(view).build();
                        sr = searcher.search(sf, efSearch, Bits.ALL);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    sr = GraphSearcher.search(queryVector, efSearch, cs.ds.getBaseRavv(), cs.ds.similarityFunction, cs.index, Bits.ALL);
                }

                var gt = cs.ds.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum());
    }

    static class ConfiguredSystem {
        DataSet ds;
        GraphIndex index;
        CompressedVectors cv;

        ConfiguredSystem(DataSet ds, GraphIndex index, CompressedVectors cv) {
            this.ds = ds;
            this.index = index;
            this.cv = cv;
        }

        public SearchScoreProvider scoreProviderFor(VectorFloat<?> queryVector, GraphIndex.View view) {
            ScoreFunction.ApproximateScoreFunction asf;
            if (index instanceof CachingADCGraphIndex) {
                asf =  ((CachingADCGraphIndex.CachedView) view).approximateScoreFunctionFor(queryVector, ds.similarityFunction);
            } else {
                asf = cv.scoreFunctionFor(queryVector, ds.similarityFunction);
            }
            var ravv = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length());
            var rr = ScoreFunction.ExactScoreFunction.from(queryVector, ds.similarityFunction, ravv);
            return new SearchScoreProvider(asf, rr);
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
}
