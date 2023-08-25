package com.github.jbellis.jvector.example;

import com.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import com.github.jbellis.jvector.example.util.SiftLoader;
import com.github.jbellis.jvector.graph.ConcurrentHnswGraphBuilder;
import com.github.jbellis.jvector.graph.HnswGraphSearcher;
import com.github.jbellis.jvector.graph.NeighborQueue;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class SiftSmall {

    public static double testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth) throws IOException, InterruptedException, ExecutionException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);

        var start = System.nanoTime();
        var builder = new ConcurrentHnswGraphBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100);
        int buildThreads = 8;
        var es = Executors.newFixedThreadPool(buildThreads);
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get();
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var topKfound = new AtomicInteger(0);
        var topK = 100;
        start = System.nanoTime();
        IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
            var queryVector = queryVectors.get(i);
            NeighborQueue nn;
            try {
                nn = HnswGraphSearcher.searchConcurrent(queryVector, 100, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, hnsw, null, Integer.MAX_VALUE);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            var gt = groundTruth.get(i);
            var n = IntStream.range(0, topK).filter(j -> gt.contains(nn.nodes()[j])).count();
            topKfound.addAndGet((int) n);
        });
        System.out.printf("  Querying %d vectors in parallel took %s seconds%n", queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        return (double) topKfound.get() / (queryVectors.size() * topK);
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        var props = loadProperties("project.properties");
        var siftPath = props.getProperty("sift.path");
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        var recall = testRecall(baseVectors, queryVectors, groundTruth);

        System.out.println("Recall: " + recall);
    }

    private static Properties loadProperties(String resourceName) throws IOException {
        Properties properties = new Properties();

        try (InputStream input = SiftSmall.class.getClassLoader().getResourceAsStream(resourceName)) {
            if (input == null) {
                throw new IOException("Resource not found: " + resourceName);
            }
            properties.load(input);
        }
        return properties;
    }
}
