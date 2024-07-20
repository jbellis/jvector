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

    private List<List<Integer>> balancedKMeansClustering(List<Integer> pointIndices, int k) {
        List<VectorFloat<?>> points = pointIndices.stream()
                .map(ravv::getVector)
                .collect(Collectors.toList());

        // Initialize centroids using k-means++
        List<VectorFloat<?>> centroids = initializeCentroids(points, k);

        // Perform balanced k-means clustering
        List<List<Integer>> clusters = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            clusters.add(new ArrayList<>());
        }

        int maxIterations = 100;
        double prevObjective = Double.MAX_VALUE;

        for (int iter = 0; iter < maxIterations; iter++) {
            // Assign points to clusters
            clusters.forEach(List::clear);
            for (int i = 0; i < points.size(); i++) {
                int bestCluster = findBestCluster(points.get(i), centroids, clusters);
                clusters.get(bestCluster).add(pointIndices.get(i));
            }

            // Update centroids
            for (int i = 0; i < k; i++) {
                if (!clusters.get(i).isEmpty()) {
                    centroids.set(i, computeCentroid(clusters.get(i)));
                }
            }

            // Calculate objective function
            double objective = calculateObjective(clusters, centroids);
            if (Math.abs(objective - prevObjective) < 1e-6) {
                break;
            }
            prevObjective = objective;
        }

        return clusters;
    }

    private List<VectorFloat<?>> initializeCentroids(List<VectorFloat<?>> points, int k) {
        List<VectorFloat<?>> centroids = new ArrayList<>(k);
        var random = new Random();

        // Choose first centroid randomly
        centroids.add(points.get(random.nextInt(points.size())));

        // Choose remaining centroids
        for (int i = 1; i < k; i++) {
            double[] distances = new double[points.size()];
            double totalDistance = 0;

            for (int j = 0; j < points.size(); j++) {
                double minDistance = Double.MAX_VALUE;
                for (VectorFloat<?> centroid : centroids) {
                    double distance = 1 - vsf.compare(points.get(j), centroid);
                    minDistance = Math.min(minDistance, distance);
                }
                distances[j] = minDistance * minDistance;
                totalDistance += distances[j];
            }

            double target = random.nextDouble() * totalDistance;
            int centroidIndex = 0;
            for (double distance : distances) {
                target -= distance;
                if (target <= 0) break;
                centroidIndex++;
            }

            centroids.add(points.get(centroidIndex));
        }

        return centroids;
    }

    private int findBestCluster(VectorFloat<?> point, List<VectorFloat<?>> centroids, List<List<Integer>> clusters) {
        int bestCluster = 0;
        double bestScore = Double.MAX_VALUE;

        for (int i = 0; i < centroids.size(); i++) {
            double distance = 1 - vsf.compare(point, centroids.get(i));
            double balancePenalty = balanceFactor * clusters.get(i).size() / ravv.size();
            double score = distance + balancePenalty;

            if (score < bestScore) {
                bestScore = score;
                bestCluster = i;
            }
        }

        return bestCluster;
    }

    private VectorFloat<?> computeCentroid(List<Integer> clusterIndices) {
        List<VectorFloat<?>> clusterPoints = clusterIndices.stream()
                .map(ravv::getVector)
                .collect(Collectors.toList());
        return KMeansPlusPlusClusterer.centroidOf(clusterPoints);
    }

    private double calculateObjective(List<List<Integer>> clusters, List<VectorFloat<?>> centroids) {
        double objective = 0;
        for (int i = 0; i < clusters.size(); i++) {
            for (int pointIndex : clusters.get(i)) {
                VectorFloat<?> point = ravv.getVector(pointIndex);
                objective += 1 - vsf.compare(point, centroids.get(i));
            }
            objective += balanceFactor * Math.pow(clusters.get(i).size() - ravv.size() / clusters.size(), 2);
        }
        return objective;
    }
}
