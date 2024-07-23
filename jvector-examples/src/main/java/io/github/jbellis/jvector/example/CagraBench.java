package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.graph.AcceleratedIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class CagraBench {
    private static final VectorUtilSupport vectorUtilSupport = VectorizationProvider.getInstance().getVectorUtilSupport();

    public static void main(String[] args) throws IOException {
        var mfd = DownloadHelper.maybeDownloadFvecs("cohere-english-v3-100k");
        var dataset = mfd.load();

        AcceleratedIndex.ExternalIndex cagra;
        if (Files.exists(Path.of("cohere.cagra"))) {
            cagra = vectorUtilSupport.loadCagraIndex("cohere.cagra");
            System.out.printf("Loaded index of %d nodes%n", cagra.size());
        } else {
            cagra = vectorUtilSupport.buildCagraIndex(dataset.getBaseRavv());
            cagra.save("cohere.cagra");
            System.out.printf("Created index of %d nodes%n", cagra.size());
        }
        var index = new AcceleratedIndex(cagra, q -> dataset.getBaseRavv().rerankerFor(q, VectorSimilarityFunction.EUCLIDEAN));

        // Test for recall
        int topK = 100; // Adjust as needed
        int rerankK = 200; // Adjust as needed
        int queryRuns = 1; // Adjust as needed

        ResultSummary results = null;
        for (int i = 0; i < 10; i++) {
            long start = System.nanoTime();
            results = performQueries(dataset, index, topK, rerankK, queryRuns);
            long end = System.nanoTime();
            System.out.printf("Took %.3f seconds%n", (end - start) / 1e9);
        }

        // Print results
        double recall = ((double) results.topKFound) / (queryRuns * dataset.queryVectors.size() * topK);
        System.out.printf("Query top %d/%d recall %.4f after %,d nodes visited%n",
                          topK, rerankK, recall, results.nodesVisited);
    }

    private static ResultSummary performQueries(DataSet dataset,
                                                AcceleratedIndex index,
                                                int topK,
                                                int rerankK,
                                                int queryRuns)
    {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();

        // run queryRuns on a new thread
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, dataset.queryVectors.size()).parallel().forEach(i -> {
                var sr = index.search(dataset.queryVectors.get(i), topK, rerankK);

                // Process search result
                var gt = dataset.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum());
    }

    private static long topKCorrect(int topK, SearchResult.NodeScore[] resultNodes, Set<Integer> gt) {
        var a = Arrays.stream(resultNodes).mapToInt(nodeScore -> nodeScore.node).toArray();
        int count = Math.min(resultNodes.length, topK);
        var resultSet = Arrays.stream(a, 0, count)
                .boxed()
                .collect(Collectors.toSet());
        return resultSet.stream().filter(gt::contains).count();
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
