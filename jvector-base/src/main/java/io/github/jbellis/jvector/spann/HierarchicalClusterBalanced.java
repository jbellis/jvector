package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntArrayList;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.max;

/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private static final int MAX_BRANCHING_FACTOR = 32;
    private static final int K_MEANS_ITERATIONS = ProductQuantization.K_MEANS_ITERATIONS;
    private final float INITIAL_LAMBDA = 100.0f;

    private final RandomAccessVectorValues ravv;
    private final VectorSimilarityFunction vsf;

    /**
     * Represents a node in the Hierarchical Cluster-Balanced tree.
     */
    private static class HCBNode {
        private final VectorFloat<?> centroid;
        private final List<HCBNode> children;
        private final IntArrayList pointIndexes;

        private HCBNode(List<HCBNode> children) {
            this.children = children;
            this.centroid = null;
            this.pointIndexes = null;
        }

        private HCBNode(VectorFloat<?> centroid, IntArrayList pointIndexes) {
            this.centroid = centroid;
            this.pointIndexes = pointIndexes;
            this.children = null;
        }
    }

    /**
     * Constructs a new HierarchicalClusterBalanced instance.
     *
     * @param ravv The RandomAccessVectorValues containing the vectors to cluster
     * @param vsf The VectorSimilarityFunction used to compute distances between vectors
     */
    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf)
    {
        this.ravv = ravv;
        this.vsf = vsf;
    }

    /**
     * Selects centroids using the Hierarchical Cluster-Balanced algorithm.
     *
     * @param nCentroids The number of centroids to select
     * @return IdentityHashMap of centroids to ConcurrentSet of assigned point indexes
     * (intended for further updates by caller)
     */
    public Map<VectorFloat<?>, Set<Integer>> computeInitialAssignments(int nCentroids) {
        var allIndices = IntStream.range(0, ravv.size()).collect(IntArrayList::new, IntArrayList::add, IntArrayList::addAll);
        HCBNode root = buildHCBTree(allIndices, max(1, ravv.size() / nCentroids));

        var results = new IdentityHashMap<VectorFloat<?>, Set<Integer>>();
        flatten(root, results);

        return results;
    }

    /**
     * Flattens the HCB tree into a map of centroids to assigned point indices.
     */
    private void flatten(HCBNode node, Map<VectorFloat<?>, Set<Integer>> assignments) {
        if (node.children == null) {
            assert node.pointIndexes != null;
            var s = ConcurrentHashMap.<Integer>newKeySet(node.pointIndexes.size());
            s.addAll(node.pointIndexes);
            assignments.put(node.centroid, s);
            return;
        }

        for (HCBNode child : node.children) {
            flatten(child, assignments);
        }
    }

    /**
     * Recursively builds the Hierarchical Cluster-Balanced tree.
     */
    private HCBNode buildHCBTree(IntArrayList pointIndices, int idealPointsPerCentroid) {
        if (pointIndices.size() <= idealPointsPerCentroid * 1.5) {
            var centroid = KMeansPlusPlusClusterer.centroidOf(pointIndices.stream().map(ravv::getVector).collect(Collectors.toList()));
            return new HCBNode(centroid, pointIndices);
        }

        return tryClustering(pointIndices, idealPointsPerCentroid);
    }

    private HCBNode tryClustering(IntArrayList pointIndices, int idealPointsPerCentroid) {
        int k = Math.min(MAX_BRANCHING_FACTOR, pointIndices.size() / idealPointsPerCentroid);
        var clusterer = new KMeansPlusPlusBalancedClusterer(pointIndices.toIntArray(), ravv, k, vsf, INITIAL_LAMBDA);
        clusterer.cluster(K_MEANS_ITERATIONS);

        var children = new ArrayList<HCBNode>();
        clusterer.getClusters().forEach((cluster, points) -> {
            children.add(buildHCBTree(points, idealPointsPerCentroid));
        });
        return new HCBNode(children);
    }
}
