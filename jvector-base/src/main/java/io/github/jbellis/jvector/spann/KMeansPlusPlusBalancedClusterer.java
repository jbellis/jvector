package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.vector.VectorUtil.addInPlace;
import static io.github.jbellis.jvector.vector.VectorUtil.scale;
import static io.github.jbellis.jvector.vector.VectorUtil.subInPlace;

public class KMeansPlusPlusBalancedClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // number of centroids to compute
    private final int k;

    // the points to train on
    private final RandomAccessVectorValues points;
    // the cluster each point is assigned to
    private final int[] assignments;
    // the centroids of each cluster
    private final VectorFloat<?>[] centroids;

    // used to accelerate updating clusters by unweighted L2 distance.
    private final int[] centroidDenoms; // the number of points assigned to each cluster
    private final VectorFloat<?>[] centroidNums; // the sum of all points assigned to each cluster

    // comparison function
    private final VectorSimilarityFunction vsf;

    // Lambda parameter for balanced clustering
    private float lambda;

    /**
     * Constructs a KMeansPlusPlusBalancedClusterer with the specified points and number of clusters.
     *
     * @param points the points to cluster
     * @param k      number of clusters
     * @param vsf    vector similarity function
     */
    public KMeansPlusPlusBalancedClusterer(RandomAccessVectorValues points, int k, VectorSimilarityFunction vsf) {
        this(points, chooseInitialCentroids(points, k, vsf), vsf);
    }

    /**
     * Constructs a KMeansPlusPlusBalancedClusterer with the specified points and initial centroids.
     *
     * @param points    the points to cluster
     * @param centroids the initial centroids
     * @param vsf       vector similarity function
     */
    public KMeansPlusPlusBalancedClusterer(RandomAccessVectorValues points, VectorFloat<?> centroids, VectorSimilarityFunction vsf) {
        this.points = points;
        this.k = centroids.length() / points.dimension();
        this.centroids = centroids.copy();
        this.vsf = vsf;
        this.lambda = 0;

        centroidDenoms = new int[k];
        centroidNums = new VectorFloat<?>[k];
        for (int i = 0; i < centroidNums.length; i++) {
            centroidNums[i] = vectorTypeSupport.createFloatVector(points.dimension());
        }
        assignments = new int[points.size()];

        initializeAssignedPoints();
    }

    public void setLambda(float lambda) {
        this.lambda = lambda;
    }

    /**
     * Refines the lambda value based on the current clustering results.
     */
    public float getRefinedLambda() {
        int maxCluster = IntStream.range(0, k).reduce((a, b) -> centroidDenoms[a] > centroidDenoms[b] ? a : b).orElse(-1);
        if (maxCluster == -1) {
            return lambda;
        }

        int dimension = points.dimension();
        VectorFloat<?> maxCenter = vectorTypeSupport.createFloatVector(dimension);
        maxCenter.copyFrom(centroids, maxCluster * dimension, 0, dimension);
        DoubleAdder totalDist = new DoubleAdder();
        AtomicInteger count = new AtomicInteger();

        IntStream.range(0, points.size()).parallel().forEach(i -> {
            VectorFloat<?> point = points.getVector(i);
            if (getNearestCluster(point) == maxCluster) {
                totalDist.add(vsf.compare(maxCenter, point));
                count.incrementAndGet();
            }
        });
        double avgDist = totalDist.doubleValue() / count.get();

        // Adjust lambda based on the average distance in the largest cluster
        return (float) ((getMaxClusterDist(maxCluster) - avgDist) / points.size());
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @param iterations number of iterations to perform
     * @return a VectorFloat of cluster centroids.
     */
    public VectorFloat<?> cluster(int iterations) {
        for (int i = 0; i < iterations; i++) {
            int changedCount = clusterOnce();
            if (changedCount <= 0.01 * points.size()) {
                break;
            }
        }
        return centroids;
    }

    public int clusterOnce() {
        updateCentroids();
        return updateAssignedPoints();
    }

    /**
     * Chooses the initial centroids for clustering.
     */
    private static VectorFloat<?> chooseInitialCentroids(RandomAccessVectorValues points, int k, VectorSimilarityFunction vsf) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.size()) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.size()));
        }

        var random = ThreadLocalRandom.current();
        VectorFloat<?> centroids = vectorTypeSupport.createFloatVector(k * points.dimension());

        float[] distances = new float[points.size()];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        VectorFloat<?> firstCentroid = points.getVector(random.nextInt(points.size()));
        centroids.copyFrom(firstCentroid, 0, 0, firstCentroid.length());
        for (int i = 0; i < points.size(); i++) {
            float distance1 = vsf.compare(points.getVector(i), firstCentroid);
            distances[i] = Math.min(distances[i], distance1);
        }

        // For each subsequent centroid
        for (int i = 1; i < k; i++) {
            float totalDistance = 0;
            for (float distance : distances) {
                totalDistance += distance;
            }

            float r = random.nextFloat() * totalDistance;
            int selectedIdx = -1;
            for (int j = 0; j < distances.length; j++) {
                r -= distances[j];
                if (r < 1e-6) {
                    selectedIdx = j;
                    break;
                }
            }

            if (selectedIdx == -1) {
                selectedIdx = random.nextInt(points.size());
            }

            VectorFloat<?> nextCentroid = points.getVector(selectedIdx);
            centroids.copyFrom(nextCentroid, 0, i * nextCentroid.length(), nextCentroid.length());

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.size(); j++) {
                float newDistance = vsf.compare(points.getVector(j), nextCentroid);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }
        return centroids;
    }

    /**
     * Assigns points to the nearest cluster. The results are stored as ordinals in `assignments`.
     * This method should only be called once after initial centroids are chosen.
     */
    private void initializeAssignedPoints() {
        for (int i = 0; i < points.size(); i++) {
            VectorFloat<?> point = points.getVector(i);
            var newAssignment = getNearestCluster(point);
            centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
            addInPlace(centroidNums[newAssignment], point);
            assignments[i] = newAssignment;
        }
    }

    /**
     * Assigns points to the nearest cluster. The results are stored as ordinals in `assignments`,
     * and `centroidNums` and `centroidDenoms` are updated to accelerate centroid recomputation.
     *
     * @return the number of points that changed clusters
     */
    private int updateAssignedPoints() {
        int changedCount = 0;

        for (int i = 0; i < points.size(); i++) {
            VectorFloat<?> point = points.getVector(i);
            var oldAssignment = assignments[i];
            var newAssignment = getNearestCluster(point);

            if (newAssignment != oldAssignment) {
                centroidDenoms[oldAssignment] = centroidDenoms[oldAssignment] - 1;
                subInPlace(centroidNums[oldAssignment], point);
                centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
                addInPlace(centroidNums[newAssignment], point);
                assignments[i] = newAssignment;
                changedCount++;
            }
        }

        return changedCount;
    }

    /**
     * Return the index of the closest centroid to the given point
     */
    private int getNearestCluster(VectorFloat<?> point) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < k; i++) {
            float distance = vsf.compare(point, centroids.slice(i * point.length(), (i + 1) * point.length()))
                    + lambda * centroidDenoms[i];
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    /**
     * Calculates centroids from centroidNums/centroidDenoms updated during point assignment
     */
    private void updateCentroids() {
        for (int i = 0; i < k; i++) {
            var denom = centroidDenoms[i];
            if (denom == 0) {
                // no points assigned to this cluster
                initializeCentroidToRandomPoint(i);
            } else {
                var centroid = centroidNums[i].copy();
                scale(centroid, 1.0f / centroidDenoms[i]);
                centroids.copyFrom(centroid, 0, i * centroid.length(), centroid.length());
            }
        }
    }

    private void initializeCentroidToRandomPoint(int i) {
        var random = ThreadLocalRandom.current();
        centroids.copyFrom(points.getVector(random.nextInt(points.size())), 0, i * points.dimension(), points.dimension());
    }

    public VectorFloat<?> getCentroids() {
        return centroids;
    }

    /**
     * @return the largest distance between any point and the centroid, for the given cluster
     */
    private float getMaxClusterDist(int cluster) {
        float maxDist = 0;
        for (int i = 0; i < points.size(); i++) {
            if (assignments[i] == cluster) {
                float dist = vsf.compare(points.getVector(i), centroids.slice(cluster * points.dimension(), (cluster + 1) * points.dimension()));
                maxDist = Math.max(maxDist, dist);
            }
        }
        return maxDist;
    }

    // Helper methods (addInPlace, subInPlace, scale) would need to be implemented
    // or imported from a utility class
}

