package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private final RandomAccessVectorValues ravv;
    private final VectorSimilarityFunction vsf;
    private final int maxDepth;
    private final int maxBranchingFactor;
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
     * @param maxBranchingFactor The maximum number of children for each node in the tree
     * @param balanceFactor The factor used to balance cluster sizes
     */
    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf,
                                       int maxDepth, int maxBranchingFactor, float balanceFactor) {
        this.ravv = ravv;
        this.vsf = vsf;
        this.maxDepth = maxDepth;
        this.maxBranchingFactor = maxBranchingFactor;
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

        return centroids.subList(0, Math.min(centroids.size(), nCentroids))
                .stream()
                .map(ravv::getVector)
                .collect(Collectors.toList());
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

        if (depth == maxDepth || pointIndices.size() <= maxBranchingFactor) {
            node.dataPointIndices = pointIndices;
            node.centroidIndex = computeCentroidIndex(pointIndices);
            return node;
        }

        int k = Math.min(maxBranchingFactor, pointIndices.size() / 2);
        List<List<Integer>> clusters = balancedKMeansClustering(pointIndices, k);

        node.centroidIndex = computeCentroidIndex(pointIndices);
        node.children = new ArrayList<>(k);

        for (List<Integer> cluster : clusters) {
            if (!cluster.isEmpty()) {
                node.children.add(buildHCBTree(cluster, depth + 1));
            }
        }

        return node;
    }
}
