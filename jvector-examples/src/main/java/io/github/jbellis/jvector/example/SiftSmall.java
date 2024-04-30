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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.Feature;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.graph.disk.InlineVectors;
import io.github.jbellis.jvector.graph.disk.LVQ;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndexWriter;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExceptionUtils;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class SiftSmall {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void siftInMemory(ArrayList<VectorFloat<?>> baseVectors) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);

            // search for a random vector
            VectorFloat<?> q = randomVector(originalDimension);
            SearchResult sr = GraphSearcher.search(q, 10, ravv, VectorSimilarityFunction.EUCLIDEAN, index, Bits.ALL);
            for (SearchResult.NodeScore ns : sr.getNodes()) {
                System.out.println(ns);
            }
        }
    }

    public static void siftInMemoryWithSearcher(ArrayList<VectorFloat<?>> baseVectors) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);

            // search for a random vector
            VectorFloat<?> q = randomVector(originalDimension);
            try (GraphSearcher searcher = new GraphSearcher(index)) {
                SearchScoreProvider ssp = SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
                SearchResult sr = searcher.search(ssp, 10, Bits.ALL);
                for (SearchResult.NodeScore ns : sr.getNodes()) {
                    System.out.println(ns);
                }
            }
        }
    }

    public static void siftInMemoryWithRecall(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
            testRecallInternal(index, queryVectors, groundTruth, sspFactory);
        }
    }

    public static void siftPersistedInline(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder builder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = builder.build(ravv);
            // write the index to disk with default options
            OnDiskGraphIndex.write(index, ravv, Files.createTempFile("siftsmall", ".inline"));
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
            testRecallInternal(index, queryVectors, groundTruth, sspFactory);
        }
    }

    public static void siftPersistedInline2(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder graphBuilder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = graphBuilder.build(ravv);
            // write the index to disk with explicit default options
            Path path = Files.createTempFile("siftsmall", ".inline");
            OnDiskGraphIndexWriter.Builder writerBuilder =
                    new OnDiskGraphIndexWriter.Builder(index, path).with(new InlineVectors(ravv.dimension()));
            try (OnDiskGraphIndexWriter writer = writerBuilder.build())
            {
                EnumMap<FeatureId, IntFunction<Feature.State>> suppliers =
                        Feature.singleStateFactory(FeatureId.INLINE_VECTORS, nodeId -> new InlineVectors.State(ravv.getVector(nodeId)));
                writer.write(suppliers);
            }
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
            testRecallInternal(index, queryVectors, groundTruth, sspFactory);
        }
    }

    public static void siftPersistedLVQ(List<VectorFloat<?>> baseVectors, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth) throws IOException {
        // infer the dimensionality from the first vector
        int originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // build an index from all of the base vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);
        try (GraphIndexBuilder graphBuilder = new GraphIndexBuilder(bsp, ravv.dimension(), 16, 100, 1.2f, 1.2f)) {
            // build the index (in memory)
            OnHeapGraphIndex index = graphBuilder.build(ravv);
            // create the LVQ compressor
            LocallyAdaptiveVectorQuantization compressor = LocallyAdaptiveVectorQuantization.compute(ravv);
            // write the index to disk with LVQ compression
            Path path = Files.createTempFile("siftsmall", ".inline");
            OnDiskGraphIndexWriter.Builder writerBuilder =
                    new OnDiskGraphIndexWriter.Builder(index, path).with(new LVQ(compressor));
            try (OnDiskGraphIndexWriter writer = writerBuilder.build())
            {
                // given the ordinal of a graph node corresponding to our source vectors, explain how to compress it
                EnumMap<FeatureId, IntFunction<Feature.State>> suppliers =
                        Feature.singleStateFactory(FeatureId.LVQ, ordinal -> new LVQ.State(compressor.encode(ravv.getVector(ordinal))));
                writer.write(suppliers);
            }
            // measure our recall against the (exactly computed) ground truth
            Function<VectorFloat<?>, SearchScoreProvider> sspFactory = q -> SearchScoreProvider.exact(q, VectorSimilarityFunction.EUCLIDEAN, ravv);
            testRecallInternal(index, queryVectors, groundTruth, sspFactory);
        }
    }

    //
    // Utilities and main() harness
    //

    public static VectorFloat<?> randomVector(int dim) {
        var R = ThreadLocalRandom.current();
        var vec = vts.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, R.nextFloat());
            if (R.nextBoolean()) {
                vec.set(i, -vec.get(i));
            }
        }
        VectorUtil.l2normalize(vec);
        return vec;
    }

    private static void testRecallInternal(GraphIndex graph, List<VectorFloat<?>> queryVectors, List<Set<Integer>> groundTruth, Function<VectorFloat<?>, SearchScoreProvider> sspFactory) throws IOException {
        AtomicInteger topKfound = new AtomicInteger(0);
        int topK = 100;
        String graphType = graph.getClass().getSimpleName();
        long start = System.nanoTime();
        try (ExplicitThreadLocal<GraphSearcher> searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(graph))) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                try (GraphSearcher searcher = searchers.get()) {
                    SearchScoreProvider ssp = sspFactory.apply(queryVector);
                    SearchResult.NodeScore[] nn = searcher.search(ssp, 100, Bits.ALL).getNodes();

                    Set<Integer> gt = groundTruth.get(i);
                    long n = IntStream.range(0, topK).filter(j -> gt.contains(nn[j].node)).count();
                    topKfound.addAndGet((int) n);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        } catch (Exception e) {
            ExceptionUtils.throwIoException(e);
        }
        System.out.printf("  (%s) Querying %d vectors in parallel took %s seconds%n", graphType, queryVectors.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        System.out.printf("(%s) Recall: %.4f%n", graphType, (double) topKfound.get() / (queryVectors.size() * topK));
    }

    public static void main(String[] args) throws IOException {
        var siftPath = "siftsmall";
        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        var groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                          baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());

        siftInMemory(baseVectors);
        siftInMemoryWithSearcher(baseVectors);
        siftInMemoryWithRecall(baseVectors, queryVectors, groundTruth);
        siftPersistedInline(baseVectors, queryVectors, groundTruth);
        siftPersistedInline2(baseVectors, queryVectors, groundTruth);
        siftPersistedLVQ(baseVectors, queryVectors, groundTruth);
    }
}
