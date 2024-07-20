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
import io.github.jbellis.jvector.vector.ArrayVectorFloat;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.Int2ObjectHashMap;
import org.agrona.collections.IntArrayList;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class IVFIndex {
    private static final double centroidFraction = 0.16; // TODO tune this for GPU

    private final GraphIndex index;
    private final RandomAccessVectorValues ravv;
    private final Int2ObjectHashMap<int[]> postings;
    private final ExplicitThreadLocal<GraphSearcher> searchers;
    private final VectorSimilarityFunction vsf;

    // Assign vector to the lists of all centroids within this factor of the closest
    private static final float CLOSURE_THRESHOLD = 0.1f; // TODO compute something based on distance sampling
    // Maximum number of assignments for each vector
    private static final int MAX_ASSIGNMENTS = 8;

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
        var centroids = hcb.computeInitialAssignments((int) (ravv.size() * centroidFraction));
        System.out.printf("%d initial centroids computed in %fs%n", centroids.size(), (System.nanoTime() - start) / 1_000_000_000.0);

        var postingsMap = new ConcurrentHashMap<Integer, Set<Integer>>(); // centroid indexes -> point indexes
        OnHeapGraphIndex index;

        int pass = 0;
        var eliminatedCentroids = new ConcurrentSkipListSet<Integer>();
        while (true) {
            postingsMap.clear();

            start = System.nanoTime();
            // Build the graph index using these centroids
            centroids = IntStream.range(0, centroids.size()).filter(i1 -> !eliminatedCentroids.contains(i1)).mapToObj(centroids::get).collect(Collectors.toList());
            try (var builder = new GraphIndexBuilder(ravv, vsf, 32, 100, 1.2f, 1.2f)) {
                index = builder.build(new ListRandomAccessVectorValues(centroids, ravv.dimension()));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            System.out.printf("Pass %d graph index with %d centroids built in %fs%n", pass, centroids.size(), (System.nanoTime() - start) / 1_000_000_000.0);

            start = System.nanoTime();
            // Assign vectors to centroids using the index
            var finalIndex = index;
            var finalCentroids = centroids;
            ThreadLocal<GraphSearcher> searchers = ThreadLocal.withInitial(() -> new GraphSearcher(finalIndex));
            IntStream.range(0, ravv.size()).parallel().forEach(i -> {
                var v = ravv.getVector(i);
                var ssp = SearchScoreProvider.exact(v, vsf, ravv);
                var sr = searchers.get().search(ssp, 2 * MAX_ASSIGNMENTS, Bits.ALL);
                assignVectorToClusters(ravv, vsf, i, sr, finalCentroids, postingsMap);
            });
            var totalAssignments = IntStream.range(0, centroids.size()).mapToLong(i -> postingsMap.getOrDefault(i, Set.of()).size()).sum();
            System.out.printf("Pass %d with %d total vector assignments in %fs%n", pass, totalAssignments, (System.nanoTime() - start) / 1_000_000_000.0);

            AtomicInteger tooSmall = new AtomicInteger();
            AtomicInteger tooLarge = new AtomicInteger();
            eliminatedCentroids.clear();
            var newCentroids = new ConcurrentSkipListSet<VectorFloat<?>>((a, b) -> {
                for (int i = 0; i < a.length(); i++) {
                    if (a.get(i) < b.get(i)) {
                        return -1;
                    } else if (a.get(i) > b.get(i)) {
                        return 1;
                    }
                }
                return 0;
            });
            var idealAssignments = 1 / centroidFraction;
            IntStream.range(0, centroids.size()).parallel().forEach(i -> {
                if (!postingsMap.containsKey(i) || postingsMap.get(i).size() < 0.5 * idealAssignments) {
                    tooSmall.incrementAndGet();
                    eliminatedCentroids.add(i);
                    return;
                }
                if (postingsMap.get(i).size() > 1.5 * idealAssignments) {
                    tooLarge.incrementAndGet();
                    eliminatedCentroids.add(i);
                    var subTree = HierarchicalClusterBalanced.createClusteredTree(ravv, toIntArrayList(postingsMap.get(i)));
                    newCentroids.addAll(subTree.flatten());
                }
            });
            printHistogram(postingsMap);
            System.out.printf("After pass %d, %d centroids too small and %d too large. adding %d new ones%n",
                              pass, tooSmall.get(), tooLarge.get(), newCentroids.size());
            pass++;
            if (eliminatedCentroids.size() + newCentroids.size() < 0.01 * centroids.size()) {
                break;
            }
            centroids.addAll(newCentroids);
        }

        var optimizedPostings = new Int2ObjectHashMap<int[]>();
        for (int i = 0; i < centroids.size(); i++) {
            optimizedPostings.put(i, postingsMap.get(i).stream().mapToInt(Integer::intValue).toArray());
        }

        return new IVFIndex(index, ravv, optimizedPostings, vsf);
    }

    private static IntArrayList toIntArrayList(Set<Integer> integers) {
        var ial = new IntArrayList(integers.size(), Integer.MIN_VALUE);
        ial.addAll(integers);
        return ial;
    }

    private static void printHistogram(Map<Integer, Set<Integer>> postings) {
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

    private static void assignVectorToClusters(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, int vectorId, SearchResult sr, List<VectorFloat<?>> centroidsList, Map<Integer, Set<Integer>> postings) {
        float baseScore = sr.getNodes()[0].score;
        List<SearchResult.NodeScore> assignedCentroids = new ArrayList<>();

        outer:
        for (int i = 0; i < sr.getNodes().length && assignedCentroids.size() < MAX_ASSIGNMENTS; i++) {
            SearchResult.NodeScore ns = sr.getNodes()[i];
            if (ns.score >= baseScore * (1 + CLOSURE_THRESHOLD)) {
                break;
            }

            // RNG rule is like HNSW diversity:
            // Before assigning a vector to a centroid, we check if there's any previously assigned centroid
            // that's closer to both the vector and the candidate centroid.  If so, we skip the assignment.
            for (SearchResult.NodeScore assigned : assignedCentroids) {
                float distanceToVector = vsf.compare(centroidsList.get(assigned.node), ravv.getVector(vectorId));
                float distanceToCandidate = vsf.compare(centroidsList.get(ns.node), ravv.getVector(vectorId));
                if (distanceToVector < distanceToCandidate && distanceToVector < ns.score) {
                    continue outer;
                }
            }

            postings.computeIfAbsent(ns.node, __ -> new ConcurrentSkipListSet<>()).add(vectorId);
            assignedCentroids.add(ns);
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
