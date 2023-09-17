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
import io.github.jbellis.jvector.disk.CompressedVectors;
import io.github.jbellis.jvector.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.example.util.ReaderSupplierFactory;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class SiftSmall {

    public static void testRecall(ArrayList<float[]> baseVectors, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, Path testDirectory) throws IOException, InterruptedException, ExecutionException {
        int originalDimension = baseVectors.get(0).length;
        var ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        var start = System.nanoTime();
        var pqDims = originalDimension / 2;
        ProductQuantization pq = ProductQuantization.compute(new ListRandomAccessVectorValues(baseVectors, originalDimension), pqDims, false);
        System.out.format("  PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(baseVectors);
        System.out.format("  PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var compressedVectors = new CompressedVectors(pq, quantizedVectors);

        start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100, 1.5f, 1.4f);
        var onHeapGraph = builder.build();
        System.out.printf("  Building index took %s seconds%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var graphPath = testDirectory.resolve("graph_test");
        try {
            DataOutputStream outputFile = new DataOutputStream(new FileOutputStream(graphPath.toFile()));
            OnDiskGraphIndex.write(onHeapGraph, ravv, outputFile);

            var onDiskGraph = new CachingGraphIndex(new OnDiskGraphIndex<>(ReaderSupplierFactory.open(graphPath), 0));

            testRecallInternal(onHeapGraph, ravv, queryVectors, groundTruth, null);
            testRecallInternal(onDiskGraph, null, queryVectors, groundTruth, compressedVectors);
        } finally {
            Files.deleteIfExists(graphPath);
        }
    }

    private static void testRecallInternal(GraphIndex<float[]> graph, RandomAccessVectorValues<float[]> ravv, ArrayList<float[]> queryVectors, ArrayList<HashSet<Integer>> groundTruth, CompressedVectors compressedVectors) {
        var topKfound = new AtomicInteger(0);
        var topK = 100;
        var graphType = graph.getClass().getSimpleName();
        var start = System.nanoTime();
        IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
            var queryVector = queryVectors.get(i);
            SearchResult.NodeScore[] nn;
            var view = graph.getView();
            var searcher = new GraphSearcher.Builder(view).build();
            if (compressedVectors == null) {
                NeighborSimilarity.ExactScoreFunction sf = (j) -> VectorSimilarityFunction.EUCLIDEAN.compare(queryVector, ravv.vectorValue(j));
                nn = searcher.search(sf, null, 100, null).getNodes();
            }
            else {
                NeighborSimilarity.ApproximateScoreFunction sf = compressedVectors.approximateScoreFunctionFor(queryVector, VectorSimilarityFunction.EUCLIDEAN);
                NeighborSimilarity.ReRanker<float[]> rr = (j, vectors) -> VectorSimilarityFunction.EUCLIDEAN.compare(queryVector, vectors.get(j));
                nn = searcher.search(sf, rr, 100, null).getNodes();
            }

            var gt = groundTruth.get(i);
            var n = IntStream.range(0, topK).filter(j -> gt.contains(nn[j].node)).count();
            topKfound.addAndGet((int) n);
        });
        System.out.printf("  (%s) Querying %d vectors in parallel took %s seconds%n", graphType, queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        System.out.printf("(%s) Recall: %.4f%n", graphType, (double) topKfound.get() / (queryVectors.size() * topK));
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        var siftPath = "siftsmall";
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);

        var testDirectory = Files.createTempDirectory("SiftSmallGraphDir");
        try {
            testRecall(baseVectors, queryVectors, groundTruth, testDirectory);
        } finally {
            Files.delete(testDirectory);
        }
    }
}
