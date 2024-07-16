package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.MMapReader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.PreloadedGraphIndex;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Demo {

    public static void main(String[] args) throws IOException {
        // Compute for the cohere-100k dataset
        var mfd = DownloadHelper.maybeDownloadFvecs("cohere-english-v3-100k");
        var dataset = mfd.load();

        // Load cohere test set.  These were written with a hacked Bench
        var pqv = PQVectors.load(new SimpleReader(Path.of("/home/jonathan/Projects/cuvs/cohere.pqv")));
        var rs = new MMapReader.Supplier(Path.of("/home/jonathan/Projects/jvector/cohere.ann"));
        var index = PreloadedGraphIndex.load(rs, 0);

        assert index.size() == pqv.count();
        System.out.printf("Loaded index of %d nodes%n", index.size());

        // Test for recall
        int topK = 100; // Adjust as needed
        int rerankK = 200; // Adjust as needed
        int queryRuns = 1; // Adjust as needed

        ResultSummary results = null;
        for (int i = 0; i < 10; i++) {
            long start = System.nanoTime();
            results = performQueries(dataset, index, pqv, topK, rerankK, queryRuns);
            long end = System.nanoTime();
            System.out.printf("Took %.3f seconds%n", (end - start) / 1e9);
        }

        // Print results
        double recall = ((double) results.topKFound) / (queryRuns * dataset.queryVectors.size() * topK);
        System.out.printf("Query top %d/%d recall %.4f after %,d nodes visited%n",
                          topK, rerankK, recall, results.nodesVisited);
    }

    private static ResultSummary performQueries(DataSet dataset,
                                                GraphIndex index,
                                                CompressedVectors cv,
                                                int topK,
                                                int rerankK,
                                                int queryRuns)
    {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();

        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, dataset.queryVectors.size()).parallel().forEach(i -> {
                VectorFloat<?> queryVector = dataset.queryVectors.get(i);
                var searcher = new GraphSearcher(index);
                var sf = createScoreProvider(queryVector, dataset, cv, searcher.getView());
                var sr = searcher.search(sf, topK, rerankK, 0.0f, 0.0f, Bits.ALL);

                // Process search result
                var gt = dataset.groundTruth.get(i);
                var n = topKCorrect(topK, sr.getNodes(), gt);
                topKfound.add(n);
                nodesVisited.add(sr.getVisitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), nodesVisited.sum());
    }

    private static SearchScoreProvider createScoreProvider(VectorFloat<?> queryVector,
                                                           DataSet dataset,
                                                           CompressedVectors cv,
                                                           GraphIndex.View view)
    {
        var scoringView = (GraphIndex.ScoringView) view;
        ScoreFunction.ApproximateScoreFunction asf = cv.precomputedScoreFunctionFor(queryVector, VectorSimilarityFunction.DOT_PRODUCT);
        var rr = scoringView.rerankerFor(queryVector, dataset.similarityFunction);
        return new SearchScoreProvider(asf, rr);
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