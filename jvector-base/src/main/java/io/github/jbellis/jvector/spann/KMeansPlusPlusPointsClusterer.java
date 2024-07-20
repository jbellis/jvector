package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntArrayList;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

import static io.github.jbellis.jvector.vector.VectorUtil.addInPlace;
import static io.github.jbellis.jvector.vector.VectorUtil.scale;
import static io.github.jbellis.jvector.vector.VectorUtil.squareL2Distance;
import static io.github.jbellis.jvector.vector.VectorUtil.subInPlace;

public class KMeansPlusPlusPointsClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // number of centroids to compute
    private final int k;

    // the points to train on
    private final int[] points;
    // source of vectors corresponding to points
    private final RandomAccessVectorValues ravv;
    // the cluster each point is assigned to
    private final int[] assignments;
    // the centroids of each cluster
    private final VectorFloat<?>[] centroids;

    // used to accelerate updating clusters by unweighted L2 distance.
    private final int[] centroidDenoms; // the number of points assigned to each cluster
    private final VectorFloat<?>[] centroidNums; // the sum of all points assigned to each cluster

    /**
     * Constructs a KMeansPlusPlusBalancedClusterer with the specified points and initial centroids.
     *
     * @param points the indexes of points to cluster
     * @param ravv   the source of vectors corresponding to points
     */
    public KMeansPlusPlusPointsClusterer(int[] points, RandomAccessVectorValues ravv, int k) {
        this.ravv = ravv;
        this.points = points;
        this.k = k;
        this.centroids = chooseInitialCentroids(points, ravv, k);

        centroidDenoms = new int[k];
        centroidNums = new VectorFloat<?>[k];
        for (int i = 0; i < k; i++) {
            centroidNums[i] = vectorTypeSupport.createFloatVector(ravv.dimension());
        }
        assignments = new int[points.length];

        initializeAssignedPoints();
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @param iterations number of iterations to perform
     */
    public void cluster(int iterations) {
        for (int i = 0; i < iterations; i++) {
            int changedCount = clusterOnce();
            if (changedCount <= 0.01 * points.length) {
                break;
            }
        }
    }

    public int clusterOnce() {
        updateCentroids();
        return updateAssignedPoints();
    }

    /**
     * Chooses the initial centroids for clustering.
     */
    private static VectorFloat<?>[] chooseInitialCentroids(int[] points, RandomAccessVectorValues ravv, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.length) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
        }

        var random = ThreadLocalRandom.current();
        VectorFloat<?>[] centroids = new VectorFloat<?>[k];

        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        centroids[0] = ravv.getVector(points[random.nextInt(points.length)]).copy();
        for (int i = 0; i < points.length; i++) {
            float distance1 = squareL2Distance(ravv.getVector(points[i]), centroids[0]);
            distances[i] = distance1;
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
                selectedIdx = random.nextInt(points.length);
            }

            centroids[i] = ravv.getVector(points[selectedIdx]).copy();

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.length; j++) {
                float newDistance = squareL2Distance(ravv.getVector(points[j]), centroids[i]);
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
        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = ravv.getVector(points[i]);
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

        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = ravv.getVector(points[i]);
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
            float distance = squareL2Distance(point, centroids[i]);
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
                centroids[i] = centroid;
            }
        }
    }

    private void initializeCentroidToRandomPoint(int i) {
        var random = ThreadLocalRandom.current();
        centroids[i] = ravv.getVector(points[random.nextInt(points.length)]).copy();
    }

    public Map<VectorFloat<?>, IntArrayList> getClusters() {
        var clusters = new IdentityHashMap<VectorFloat<?>, IntArrayList>();
        for (int i = 0; i < points.length; i++) {
            clusters.computeIfAbsent(centroids[assignments[i]], __ -> new IntArrayList()).add(points[i]);
        }
        return clusters;
    }
}

