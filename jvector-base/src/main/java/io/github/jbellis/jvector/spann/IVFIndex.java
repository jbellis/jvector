package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.ExplicitThreadLocal;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class IVFIndex {
    private static final double centroidFraction = 0.16; // TODO tune this for GPU

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

    public static IVFIndex build(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf) {
        long start = System.nanoTime();
        // Select centroids using HCB
        HierarchicalClusterBalanced hcb = new HierarchicalClusterBalanced(ravv, vsf);
        Map<VectorFloat<?>, Set<Integer>> postings = hcb.computeInitialAssignments((int) (ravv.size() * centroidFraction));
        System.out.printf("%d centroids computed in %fs%n", postings.size(), (System.nanoTime() - start) / 1_000_000_000.0);
        printHistogram(postings);

        start = System.nanoTime();
        // Build the graph index using these centroids
        OnHeapGraphIndex index;
        ArrayList<VectorFloat<?>> centroidsList;
        try (var builder = new GraphIndexBuilder(ravv, vsf, 32, 100, 1.2f, 1.2f)) {
            centroidsList = new ArrayList<>(postings.keySet());
            index = builder.build(new ListRandomAccessVectorValues(centroidsList, ravv.dimension()));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        System.out.printf("Graph index built in %fs%n", (System.nanoTime() - start) / 1_000_000_000.0);

        var optimizedPostings = new Int2ObjectHashMap<int[]>();
        for (int i = 0; i < centroidsList.size(); i++) {
            var centroid = centroidsList.get(i);
            optimizedPostings.put(i, postings.get(centroid).stream().mapToInt(Integer::intValue).toArray());
        }
        System.out.printf("Assigned vectors to centroids with closure in %ss%n", (System.nanoTime() - start) / 1_000_000_000.0);
        printHistogram(postings);

        return new IVFIndex(index, ravv, optimizedPostings, vsf);
    }

    private static void printHistogram(Map<VectorFloat<?>, Set<Integer>> postings) {
        var histogram = new int[postings.values().stream().mapToInt(s -> s.size() / 200).max().orElse(0) + 1];
        postings.values().forEach(s -> histogram[s.size() / 200]++);
        System.out.println("Histogram of postings lengths, visualized:");
        for (int i = 0; i < histogram.length; i++) {
            if (histogram[i] > 0) {
                System.out.printf("%3d: %s%n", i * 200, "*".repeat((int) Math.ceil(Math.log(histogram[i]))));
            } else {
                System.out.printf("%3d: %n", i * 200);
            }
        }
    }

    public SearchResult search(VectorFloat<?> queryVector, int topK, int nCentroids) {
        var searcher = searchers.get();
        var ssp = SearchScoreProvider.exact(queryVector, vsf, ravv);
        Set<Integer> allPostings = ConcurrentHashMap.newKeySet();
        SearchResult centroidsResult = null;
        // search until we find a non-empty centroid
        while (true) {
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
