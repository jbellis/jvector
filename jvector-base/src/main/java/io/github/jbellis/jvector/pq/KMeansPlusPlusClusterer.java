/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A KMeans++ implementation for float vectors.  Optimizes to use SIMD vector instructions if available.
 */
public class KMeansPlusPlusClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    // number of centroids to compute
    private final int k;
    // the points to train on
    private final VectorFloat<?>[] points;
    // the cluster each point is assigned to
    private final int[] assignments;
    // the centroids of each cluster
    private final VectorFloat<?> centroids;
    // the number of points assigned to each cluster
    private final int[] centroidDenoms;
    private final VectorFloat<?>[] centroidNums;

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified points and number of clusters.
     *
     * @param points the points to cluster.
     * @param k number of clusters.
     */
    public KMeansPlusPlusClusterer(VectorFloat<?>[] points, int k) {
        this(points, chooseInitialCentroids(points, k));
    }

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified points and initial centroids.
     * <p>
     * The initial centroids provided as a parameter are copied before modification.
     *
     * @param points the points to cluster.
     * @param centroids the initial centroids.
     */
    public KMeansPlusPlusClusterer(VectorFloat<?>[] points, VectorFloat<?> centroids) {
        this.points = points;
        this.k = centroids.length() / points[0].length();
        this.centroids = centroids.copy();
        centroidDenoms = new int[k];
        // initialize with empty vectors
        centroidNums = new VectorFloat<?>[k];
        for (int i = 0; i < centroidNums.length; i++) {
            centroidNums[i] = vectorTypeSupport.createFloatVector(points[0].length());
        }
        assignments = new int[points.length];

        initializeAssignedPoints();
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @return a VectorFloat of cluster centroids.
     */
    public VectorFloat<?> cluster(int maxIterations) {
        for (int i = 0; i < maxIterations; i++) {
            int changedCount = clusterOnce();
            if (changedCount <= 0.01 * points.length) {
                break;
            }
        }
        return centroids;
    }

    // This is broken out as a separate public method to allow implementing OPQ efficiently
    public int clusterOnce() {
        updateCentroids();
        return updateAssignedPoints();
    }

    /**
     * Chooses the initial centroids for clustering.
     * The first centroid is chosen randomly from the data points. Subsequent centroids
     * are selected with a probability proportional to the square of their distance
     * to the nearest existing centroid. This ensures that the centroids are spread out
     * across the data and not initialized too closely to each other, leading to better
     * convergence and potentially improved final clusterings.
     *
     * @return an array of initial centroids.
     */
    private static VectorFloat<?> chooseInitialCentroids(VectorFloat<?>[] points, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.length) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
        }

        var random = ThreadLocalRandom.current();
        VectorFloat<?> centroids = vectorTypeSupport.createFloatVector(k * points[0].length());

        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        VectorFloat<?> firstCentroid = points[random.nextInt(points.length)];
        centroids.copyFrom(firstCentroid, 0, 0, firstCentroid.length());
        for (int i = 0; i < points.length; i++) {
            float distance1 = VectorUtil.squareL2Distance(points[i], firstCentroid);
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
                selectedIdx = random.nextInt(points.length);
            }

            VectorFloat<?> nextCentroid = points[selectedIdx];
            centroids.copyFrom(nextCentroid, 0, i * nextCentroid.length(), nextCentroid.length());

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.length; j++) {
                float newDistance = VectorUtil.squareL2Distance(points[j], nextCentroid);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }
        assertFinite(centroids);
        return centroids;
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
     * This method should only be called once after initial centroids are chosen.
     */
    private void initializeAssignedPoints() {
        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = points[i];
            var newAssignment = getNearestCluster(point);
            centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
            VectorUtil.addInPlace(centroidNums[newAssignment], point);
            assignments[i] = newAssignment;
        }
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
     * This method relies on valid assignments existing from either initializeAssignedPoints or
     * a previous invocation of this method.
     *
     * @return the number of points that changed clusters
     */
    private int updateAssignedPoints() {
        int changedCount = 0;

        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = points[i];
            var oldAssignment = assignments[i];
            var newAssignment = getNearestCluster(point);

            if (newAssignment != oldAssignment) {
                centroidDenoms[oldAssignment] = centroidDenoms[oldAssignment] - 1;
                VectorUtil.subInPlace(centroidNums[oldAssignment], point);
                centroidDenoms[newAssignment] = centroidDenoms[newAssignment] + 1;
                VectorUtil.addInPlace(centroidNums[newAssignment], point);
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
            float distance = VectorUtil.squareL2Distance(point, 0, centroids, i * point.length(), point.length());
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    @SuppressWarnings({"AssertWithSideEffects", "ConstantConditions"})
    private static void assertFinite(VectorFloat<?> vector) {
        boolean assertsEnabled = false;
        assert assertsEnabled = true;

        if (assertsEnabled) {
            for (int i = 0; i < vector.length(); i++) {
                assert Float.isFinite(vector.get(i)) : "vector " + vector + " contains non-finite value";
            }
        }
    }

    /**
     * Calculates centroids from centroidNums/centroidDenoms updated during point assignment
     */
    private void updateCentroids() {
        var random = ThreadLocalRandom.current();
        for (int i = 0; i < k; i++) {
            var denom = centroidDenoms[i];
            if (denom == 0) {
                centroids.copyFrom(points[random.nextInt(points.length)], 0, i * points[0].length(), points[0].length());
            } else {
                var centroid = centroidNums[i].copy();
                VectorUtil.scale(centroid, 1.0f / centroidDenoms[i]);
                centroids.copyFrom(centroid, 0, i * centroid.length(), centroid.length());
            }
        }
    }

    /**
     * Computes the centroid of a list of points.
     */
    public static VectorFloat<?> centroidOf(List<VectorFloat<?>> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        VectorFloat<?> centroid = VectorUtil.sum(points);
        VectorUtil.scale(centroid, 1.0f / points.size());

        return centroid;
    }

    public VectorFloat<?> getCentroids() {
        return centroids;
    }
}
