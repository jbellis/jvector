package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.vector.VectorUtil.squareL2Distance;


/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final RandomAccessVectorValues ravv;
    private final VectorSimilarityFunction vsf;
    private final int maxDepth;
    private final int branchingFactor;
    private float balanceFactor;

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
     * @param balanceFactor The initial factor used to balance cluster sizes
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
     * @return A list of vectors representing the selected centroids
     */
    public List<VectorFloat<?>> computeCentroids(int nCentroids) {
        List<Integer> allIndices = IntStream.range(0, ravv.size())
                .boxed()
                .collect(Collectors.toList());

        VectorFloat<?>[] allPoints = allIndices.stream()
                .map(ravv::getVector)
                .toArray(VectorFloat<?>[]::new);

        KMeansPlusPlusClusterer initialClusterer = new KMeansPlusPlusClusterer(allPoints, nCentroids);
        VectorFloat<?> initialCentroids = initialClusterer.chooseInitialCentroids(allPoints, nCentroids);

        HCBNode root = buildHCBTree(allIndices, 0, initialCentroids);

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
     * @param initialCentroids The initial centroids to use for clustering
     * @return The constructed HCBNode
     */
    private HCBNode buildHCBTree(List<Integer> pointIndices, int depth, VectorFloat<?> initialCentroids) {
        HCBNode node = new HCBNode();

        if (depth == maxDepth || pointIndices.size() <= branchingFactor) {
            node.dataPointIndices = pointIndices;
            node.centroidIndex = computeCentroidIndex(pointIndices);
            return node;
        }

        VectorFloat<?>[] points = pointIndices.stream()
                .map(ravv::getVector)
                .toArray(VectorFloat<?>[]::new);

        KMeansPlusPlusClusterer clusterer = new KMeansPlusPlusClusterer(points, initialCentroids, KMeansPlusPlusClusterer.UNWEIGHTED);
        clusterer.cluster(ProductQuantization.K_MEANS_ITERATIONS, 0);

        double lambda = computeLambda(pointIndices.size());
        lambda = refineLambda(clusterer, points, lambda);

        node.centroidIndex = computeCentroidIndex(pointIndices);
        node.children = new ArrayList<>(branchingFactor);

        // Assign points to clusters with refined lambda
        Map<Integer, List<Integer>> clusters = new HashMap<>();
        for (int i = 0; i < points.length; i++) {
            int clusterIndex = getNearestClusterWithLambda(clusterer, points[i], lambda);
            clusters.computeIfAbsent(clusterIndex, k -> new ArrayList<>()).add(pointIndices.get(i));
        }

        // Recursively build child nodes
        for (List<Integer> cluster : clusters.values()) {
            if (!cluster.isEmpty()) {
                node.children.add(buildHCBTree(cluster, depth + 1, clusterer.getCentroids()));
            }
        }

        return node;
    }

    /**
     * Computes the initial lambda value based on the number of points.
     *
     * @param size The number of points in the current cluster
     * @return The computed lambda value
     */
    private float computeLambda(int size) {
        return balanceFactor / size;
    }

    /**
     * Refines the lambda value based on the current clustering results.
     *
     * @param clusterer The KMeansPlusPlusClusterer instance
     * @param points The points being clustered
     * @param initialLambda The initial lambda value
     * @return The refined lambda value
     */
    private double refineLambda(KMeansPlusPlusClusterer clusterer, VectorFloat<?>[] points, double initialLambda) {
        int maxCluster = IntStream.range(0, clusterer.k).reduce((a, b) -> clusterer.getClusterSize(a) > clusterer.getClusterSize(b) ? a : b).orElse(-1);
        if (maxCluster == -1) {
            return initialLambda;
        }

        VectorFloat<?> maxCenter = vts.createFloatVector(ravv.dimension());
        maxCenter.copyFrom(clusterer.getCentroids(), maxCluster * ravv.dimension(), 0, ravv.dimension());
        DoubleAdder totalDist = new DoubleAdder();
        AtomicInteger count = new AtomicInteger();
        Arrays.stream(points).parallel().forEach(point -> {
            if (clusterer.getNearestCluster(point) == maxCluster) {
                totalDist.add(vsf.compare(maxCenter, point));
                count.incrementAndGet();
            }
        });
        double avgDist = totalDist.doubleValue() / count.get();

        // Adjust lambda based on the average distance in the largest cluster
        return (clusterer.getMaxClusterDist(maxCluster) - avgDist) / points.length;
    }

    /**
     * Finds the nearest cluster for a given point, taking into account the lambda factor.
     *
     * @param clusterer The KMeansPlusPlusClusterer instance
     * @param point The point to find the nearest cluster for
     * @param lambda The lambda factor for balancing cluster sizes
     * @return The index of the nearest cluster
     */
    private int getNearestClusterWithLambda(KMeansPlusPlusClusterer clusterer, VectorFloat<?> point, double lambda) {
        int nearestCluster = -1;
        double minDistance = Float.MAX_VALUE;
        VectorFloat<?> centroids = clusterer.getCentroids();

        for (int i = 0; i < clusterer.k; i++) {
            double distance = squareL2Distance(point, 0, centroids, i * point.length(), ravv.dimension());
            distance += lambda * clusterer.getClusterSize(i);  // Add cluster size penalty
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }
        return nearestCluster;
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
