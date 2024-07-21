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
import org.agrona.collections.IntArrayList;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class IVFIndex {
    private final GraphIndex index;
    private final RandomAccessVectorValues ravv;
    private final Int2ObjectHashMap<int[]> postings;
    private final ExplicitThreadLocal<GraphSearcher> searchers;
    private final VectorSimilarityFunction vsf;

    // Maximum number of assignments for each vector
    private static final int MAX_ASSIGNMENTS = 1;

    public IVFIndex(GraphIndex index, RandomAccessVectorValues ravv, Int2ObjectHashMap<int[]> postings, VectorSimilarityFunction vsf) {
        this.index = index;
        this.ravv = ravv;
        this.postings = postings;
        searchers = ExplicitThreadLocal.withInitial(() -> new GraphSearcher(index));
        this.vsf = vsf;
    }

    public static IVFIndex build(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf, float centroidFraction) {
        long start = System.nanoTime();
        // Select centroids using HCB
        var centroids = selectRandomCentroids(ravv, centroidFraction);
        System.out.printf("%d initial centroids computed in %fs%n", centroids.size(), (System.nanoTime() - start) / 1_000_000_000.0);

        var postingsMap = new ConcurrentHashMap<Integer, Set<Integer>>(); // centroid indexes -> point indexes
        OnHeapGraphIndex index;

        float multipleAssignmentThreshold = computeDynamicThreshold(ravv, vsf);

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
                var sr = searchers.get().search(ssp, 100, Bits.ALL);
                assignVectorToClusters(ravv, vsf, i, sr, finalCentroids, postingsMap, multipleAssignmentThreshold);
            });
            var totalAssignments = IntStream.range(0, centroids.size()).mapToLong(i -> postingsMap.getOrDefault(i, Set.of()).size()).sum();
            System.out.printf("Pass %d with %d total vector assignments in %fs%n", pass, totalAssignments, (System.nanoTime() - start) / 1_000_000_000.0);
            if (true) break;
            printHistogram(postingsMap);

            float ratio = (float) totalAssignments / ravv.size();
            int idealExpandedAssignments = (int) Math.ceil(ratio / centroidFraction);
            float largeThresholdFactor = 1.5f + 0.1f * pass;
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
            IntStream.range(0, centroids.size()).parallel().forEach(i -> {
                if (!postingsMap.containsKey(i)) {
                    tooSmall.incrementAndGet();
                    eliminatedCentroids.add(i);
                    return;
                }
                if (postingsMap.get(i).size() > largeThresholdFactor * idealExpandedAssignments) {
                    tooLarge.incrementAndGet();
                    eliminatedCentroids.add(i);
                    var subPoints = toIntArray(postingsMap.get(i));
                    if (subPoints.length < 2) {
                        return;
                    }
                    var subClustering = new KMeansPlusPlusPointsClusterer(subPoints, ravv, 2);
                    subClustering.cluster(64);
                    newCentroids.addAll(List.of(subClustering.getCentroids()));
                }
            });
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
            var a = postingsMap.containsKey(i) ? postingsMap.get(i).stream().mapToInt(Integer::intValue).toArray() : new int[0];
            optimizedPostings.put(i, a);
        }

        return new IVFIndex(index, ravv, optimizedPostings, vsf);
    }

    private static int[] toIntArray(Set<Integer> integers) {
        return integers.stream().mapToInt(Integer::intValue).toArray();
    }

    private static IntArrayList toIntArrayList(Set<Integer> integers) {
        var ial = new IntArrayList(integers.size(), Integer.MIN_VALUE);
        ial.addAll(integers);
        return ial;
    }

    private static void printHistogram(Map<Integer, Set<Integer>> postings) {
        var histogram = new int[postings.values().stream().mapToInt(s -> s.size() / 200).max().orElse(0) + 1];
        postings.values().forEach(s -> histogram[s.size() / 200]++);
        // crop off zeros at the end
        int lastNonZero = histogram.length - 1;
        while (lastNonZero > 0 && histogram[lastNonZero] == 0) {
            lastNonZero--;
        }
        var cropped = Arrays.copyOf(histogram, lastNonZero + 1);
        System.out.println("Histogram of postings lengths, visualized:");
        for (int i = 0; i < cropped.length; i++) {
            if (cropped[i] > 1) {
                System.out.printf("%4d: %s %d%n", i * 10, "*".repeat((int) Math.ceil(log2(cropped[i]))), cropped[i]);
            } else if (cropped[i] == 1) {
                System.out.printf("%4d: 1%n", i * 10);
            } else {
                System.out.printf("%4d: %n", i * 10);
            }
        }
    }

    private static double log2(double a) {
        return Math.log(a) / Math.log(2);
    }

    private static float computeDynamicThreshold(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf) {
        Random R = new Random(); // Use a fixed seed for reproducibility
        int SAMPLE_SIZE = 1000;

        float[] similarities = new float[SAMPLE_SIZE * SAMPLE_SIZE];

        IntStream.range(0, SAMPLE_SIZE).parallel().forEach(i -> {
            VectorFloat<?> v1 = ravv.getVector(R.nextInt(ravv.size()));
            for (int j = 0; j < SAMPLE_SIZE; j++) {
                VectorFloat<?> v2 = ravv.getVector(R.nextInt(ravv.size()));
                float similarity = vsf.compare(v1, v2);
                similarities[i * SAMPLE_SIZE + j] = similarity;
            }
        });

        Arrays.parallelSort(similarities);
        // this cuts the number of extra assignments by about 50%
        // not sure if that's good or bad tbh
        // should we try forcing out closure assignments after the fact?  only keep the closest N points per centroid?
        int percentileIndex = (int) (0.002 * similarities.length);
        return similarities[similarities.length - percentileIndex];
    }

    private static List<VectorFloat<?>> selectRandomCentroids(RandomAccessVectorValues baseRavv, double centroidFraction) {
        // this creates redundant indexes (since we are using source vectors as centroids, each will also
        // end up assigned to itself in the posting lists)
        // we will fix this by switching to HBC centroid selection
        //
        // in the meantime: this is worse than i thought, only about 20% of the centroids get all of the vectors mapped to them
        int nCentroids = (int) (baseRavv.size() * centroidFraction);
        Set<VectorFloat<?>> selected = ConcurrentHashMap.newKeySet();
        var R = new Random();
        while (selected.size() < nCentroids) {
            selected.add(baseRavv.getVector(R.nextInt(baseRavv.size())));
        }
        return new ArrayList<>(selected);
    }

    private static List<VectorFloat<?>> selectHCBCentroids(RandomAccessVectorValues ravv, double centroidFraction) {
        int idealAssignments = (int) Math.round(1 / centroidFraction);
        HierarchicalClusterBalanced hcb = new HierarchicalClusterBalanced(ravv);
        return hcb.computeCentroids(idealAssignments);
    }

    /**
     * @return the centroids to which we will assign the rest of the vectors
     */
    private static List<VectorFloat<?>> selectKmeansCentroids(RandomAccessVectorValues baseRavv, double centroidFraction) {
        // use a zero-round KMPPPC
        int[] allPoints = IntStream.range(0, baseRavv.size()).toArray();
        var clusterer = new KMeansPlusPlusPointsClusterer(allPoints, baseRavv, (int) (baseRavv.size() * centroidFraction));
        return List.of(clusterer.getCentroids());
    }

    /**
     * Assign the given vectorId to up to the closest MAX_ASSIGNMENTS centroids
     */
    private static void assignVectorToClusters(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf,
                                               int vectorId, SearchResult sr, List<VectorFloat<?>> centroidsList,
                                               Map<Integer, Set<Integer>> postings, float threshold) {
        List<SearchResult.NodeScore> assignedCentroids = new ArrayList<>();
        assignedCentroids.add(sr.getNodes()[0]);

        outer:
        for (int i = 1; i < sr.getNodes().length && assignedCentroids.size() < MAX_ASSIGNMENTS; i++) {
            SearchResult.NodeScore ns = sr.getNodes()[i];
            if (ns.score < threshold) {
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

            assignedCentroids.add(ns);
        }

        for (var ns: assignedCentroids) {
            postings.computeIfAbsent(ns.node, __ -> ConcurrentHashMap.newKeySet()).add(vectorId);
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
            // TODO prune to centroids within top X percentile
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
        return new SearchResult(scoredPostings,
                                centroidsResult.getVisitedCount() + allPostings.size(),
                                allPostings.size(),
                                Float.POSITIVE_INFINITY);
    }
}
