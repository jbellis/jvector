package com.github.jbellis.jvector.example;

import com.github.jbellis.jvector.disk.*;
import com.github.jbellis.jvector.graph.*;
import com.github.jbellis.jvector.example.util.SiftLoader;
import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class SiftSmall {

    public static void testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, File testOutputFile) throws IOException, InterruptedException, ExecutionException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);

        var start = System.nanoTime();
        var pqDims = baseVectors.get(0).length / 2;
        ProductQuantization pq = new ProductQuantization(baseVectors, pqDims, false);
        System.out.format("  PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(baseVectors);
        System.out.format("  PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var compressedVectors = new CompressedVectors(pq, quantizedVectors);

        start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100, 1.5f, 1.4f);
        var onHeapGraph = builder.build();
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        DataOutputStream outputFile = new DataOutputStream(new FileOutputStream(testOutputFile));
        OnDiskGraphIndex.write(onHeapGraph, ravv, outputFile);

        var onDiskGraph = new OnDiskGraphIndex<float[]>(() -> {
            try {
                return new FileRandomAccessReader(testOutputFile.getAbsolutePath());
            } catch (FileNotFoundException e) {
                throw new UncheckedIOException(e);
            }
        }, 0);

        testRecallInternal(onHeapGraph, ravv, queryVectors, groundTruth, null);
        testRecallInternal(onDiskGraph, null, queryVectors, groundTruth, compressedVectors);
    }

    private static void testRecallInternal(GraphIndex<float[]> graph, RandomAccessVectorValues<float[]> ravv, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, CompressedVectors compressedVectors) {
        var topKfound = new AtomicInteger(0);
        var topK = 100;
        var graphType = graph.getClass().getSimpleName();
        var start = System.nanoTime();
        IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
            var queryVector = queryVectors.get(i);
            NeighborQueue nn;
            var view = graph.getView();
            var searcher = new GraphSearcher.Builder(view).build();
            if (compressedVectors != null) {
                NeighborSimilarity.ExactScoreFunction esf = (j) -> VectorSimilarityFunction.COSINE.compare(queryVector, view.getVector(j));
                NeighborSimilarity.ApproximateScoreFunction asf = (j) -> compressedVectors.decodedSimilarity(j, queryVector, VectorSimilarityFunction.COSINE);
                nn = searcher.search(esf, asf, 100, null, Integer.MAX_VALUE);
            } else {
                NeighborSimilarity.ExactScoreFunction scoreFunction = (j) -> VectorSimilarityFunction.COSINE.compare(queryVector, ravv.vectorValue(j));
                nn = searcher.search(scoreFunction, 100, null, Integer.MAX_VALUE);
            }

            var gt = groundTruth.get(i);
            var n = IntStream.range(0, topK).filter(j -> gt.contains(nn.nodesCopy()[j])).count();
            topKfound.addAndGet((int) n);
        });
        System.out.printf("  (%s) Querying %d vectors in parallel took %s seconds%n", graphType, queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        System.out.printf("(%s) Recall: %.4f%n", graphType, (double) topKfound.get() / (queryVectors.size() * topK));
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        var props = loadProperties("project.properties");
        var siftPath = props.getProperty("sift.path");
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        var testDirectory = Files.createTempDirectory("SiftSmallGraphDir");
        var testOutputPath = testDirectory.resolve("graph_test");
        try {
            testRecall(baseVectors, queryVectors, groundTruth, testOutputPath.toFile());
        } catch (Exception e)
        {
            e.printStackTrace();
        }
        finally {
            Files.delete(testOutputPath);
            Files.delete(testDirectory);
        }

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
