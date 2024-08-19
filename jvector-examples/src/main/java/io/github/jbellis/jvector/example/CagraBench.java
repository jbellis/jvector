package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.graph.AcceleratedIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.vector.VectorUtilSupport;
import io.github.jbellis.jvector.vector.VectorizationProvider;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

public class CagraBench {
    private static final VectorUtilSupport vectorUtilSupport = VectorizationProvider.getInstance().getVectorUtilSupport();

    public static void main(String[] args) throws IOException {
        var mfd = DownloadHelper.maybeDownloadFvecs("cohere-english-v3-100k");
        var dataset = mfd.load();

        long start = System.nanoTime();
        AcceleratedIndex.ExternalIndex cagra;
        if (Files.exists(Path.of("cohere.cagra"))) {
            cagra = vectorUtilSupport.loadCagraIndex("cohere.cagra");
            System.out.printf("Loaded index of %d nodes in %ss%n", cagra.size(), (System.nanoTime() - start) / 1e9);
        } else {
            cagra = vectorUtilSupport.buildCagraIndex(dataset.getBaseRavv());
            cagra.save("cohere.cagra");
            System.out.printf("Created index of %d nodes in %ss%n", cagra.size(), (System.nanoTime() - start) / 1e9);
        }

        // Time searches + report recall
        int topK = 100;
        ResultSummary results = null;
        for (int i = 0; i < 10; i++) {
            start = System.nanoTime();
            results = performQueries(dataset, cagra, topK);
            long end = System.nanoTime();
            System.out.printf("Took %.3f seconds%n", (end - start) / 1e9);
        }

        // Print results
        double recall = ((double) results.topKFound) / (dataset.queryVectors.size() * topK);
        System.out.printf("Query top %d/%d recall %.4f after %,d nodes visited%n",
                          topK, topK, recall, results.nodesVisited);
    }

    private static ResultSummary performQueries(DataSet dataset,
                                                AcceleratedIndex.ExternalIndex index,
                                                int topK)
    {
        LongAdder topKfound = new LongAdder();

        IntStream.range(0, dataset.queryVectors.size()).parallel().forEach(i -> {
            var sr = index.search(dataset.queryVectors.get(i), topK);

            // Process search result
            var gt = dataset.groundTruth.get(i);
            var n = topKCorrect(topK, sr, gt);
            topKfound.add(n);
        });
        return new ResultSummary((int) topKfound.sum(), -1);
    }

    private static long topKCorrect(int topK, NodesIterator resultNodes, Set<Integer> gt) {
        assert resultNodes.size() <= topK;

        long count = 0;
        while (resultNodes.hasNext()) {
            int node = resultNodes.next();
            if (gt.contains(node)) {
                count++;
            }
        }

        return count;
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
