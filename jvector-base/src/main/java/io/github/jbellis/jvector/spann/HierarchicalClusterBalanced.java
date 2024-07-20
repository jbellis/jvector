package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private final RandomAccessVectorValues ravv;
    private final VectorSimilarityFunction vsf;
    private final int maxDepth;
    private final int branchingFactor;
    private final float balanceFactor;

    /**
     * Represents a node in the Hierarchical Cluster-Balanced tree.
     */
    private static class HCBNode {
        int centroidIndex;
        List<HCBNode> children;
        List<Integer> dataPointIndices;
    }

    /**
     * Constructs a new HierarchicalClusterBalanced instance.
     *
     * @param ravv The RandomAccessVectorValues containing the vectors to cluster
     * @param vsf The VectorSimilarityFunction used to compute distances between vectors
     * @param maxDepth The maximum depth of the hierarchical clustering tree
     * @param branchingFactor The maximum number of children for each node in the tree
     * @param balanceFactor The factor used to balance cluster sizes
     */
    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf,
                                       int maxDepth, int branchingFactor, float balanceFactor) {
        this.ravv = ravv;
        this.vsf = vsf;
        this.maxDepth = maxDepth;
        this.branchingFactor = branchingFactor;
        this.balanceFactor = balanceFactor;
    }

    /**
     * Selects centroids using the Hierarchical Cluster-Balanced algorithm.
     *
     * @param nCentroids The number of centroids to select
     * @return A list of indices representing the selected centroids
     */
    public List<VectorFloat<?>> computeCentroids(int nCentroids) {
        List<Integer> allIndices = IntStream.range(0, ravv.size())
                .boxed()
                .collect(Collectors.toList());

        HCBNode root = buildHCBTree(allIndices, 0);

        List<Integer> centroids = new ArrayList<>();
        collectCentroids(root, centroids);

        var L = centroids.subList(0, Math.min(centroids.size(), nCentroids));
        return L.stream().map(ravv::getVector).collect(Collectors.toList());
    }

    /**
     * Recursively builds the Hierarchical Cluster-Balanced tree.
     *
     * @param pointIndices The indices of points to cluster at this node
     * @param depth The current depth in the tree
     * @return The constructed HCBNode
     */
    private HCBNode buildHCBTree(List<Integer> pointIndices, int depth) {
        HCBNode node = new HCBNode();

        if (depth == maxDepth || pointIndices.size() <= branchingFactor) {
            node.dataPointIndices = pointIndices;
            node.centroidIndex = computeCentroidIndex(pointIndices);
            return node;
        }

        VectorFloat<?>[] points = pointIndices.stream()
                .map(ravv::getVector)
                .toArray(VectorFloat<?>[]::new);

        KMeansPlusPlusClusterer clusterer = new KMeansPlusPlusClusterer(points, branchingFactor);
        clusterer.cluster(ProductQuantization.K_MEANS_ITERATIONS, 0); // TODO is this the right number?

        node.centroidIndex = computeCentroidIndex(pointIndices);
        node.children = new ArrayList<>(branchingFactor);

        // Assign points to clusters
        int[] assignments = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            assignments[i] = clusterer.getNearestCluster(points[i]);
        }

        // Group points by cluster
        Map<Integer, List<Integer>> clusters = new HashMap<>();
        for (int i = 0; i < assignments.length; i++) {
            clusters.computeIfAbsent(assignments[i], k -> new ArrayList<>()).add(pointIndices.get(i));
        }

        // Recursively build child nodes
        for (List<Integer> cluster : clusters.values()) {
            if (!cluster.isEmpty()) {
                node.children.add(buildHCBTree(cluster, depth + 1));
            }
        }

        return node;
    }

    /**
     * Recursively collects centroids from the HCB tree.
     *
     * @param node The current node in the tree
     * @param centroids The list to store collected centroids
     */
    private void collectCentroids(HCBNode node, List<Integer> centroids) {
        centroids.add(node.centroidIndex);
        if (node.children != null) {
            for (HCBNode child : node.children) {
                collectCentroids(child, centroids);
            }
        }
    }

    /**
     * Computes the centroid index for a given set of point indices.
     *
     * @param pointIndices The indices of points to compute the centroid for
     * @return The index of the point closest to the computed centroid
     */
    private int computeCentroidIndex(List<Integer> pointIndices) {
        List<VectorFloat<?>> points = pointIndices.stream()
                .map(ravv::getVector)
                .collect(Collectors.toList());
        VectorFloat<?> centroid = KMeansPlusPlusClusterer.centroidOf(points);
        return findNearestPointIndex(centroid, pointIndices);
    }

    /**
     * Finds the index of the point nearest to the given centroid.
     *
     * @param centroid The centroid vector
     * @param pointIndices The indices of points to search
     * @return The index of the nearest point to the centroid
     */
    private int findNearestPointIndex(VectorFloat<?> centroid, List<Integer> pointIndices) {
        int nearestIndex = -1;
        float minDistance = Float.MAX_VALUE;
        for (int index : pointIndices) {
            float distance = vsf.compare(centroid, ravv.getVector(index));
            if (distance < minDistance) {
                minDistance = distance;
                nearestIndex = index;
            }
        }
        return nearestIndex;
    }
}
