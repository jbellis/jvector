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

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A KMeans++ implementation for float vectors.  Optimizes to use SIMD vector instructions if available.
 */
public class KMeansPlusPlusClusterer {
    private final Random random;

    // number of centroids to compute
    private final int k;

    // the points to train on
    private final float[][] points;
    // the cluster each point is assigned to
    private final int[] assignments;
    // the centroids of each cluster
    private final float[][] centroids;
    // the number of points assigned to each cluster
    private final int[] centroidDenoms;
    private final float[][] centroidNums;

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified number of clusters,
     * maximum iterations, and distance function.
     *
     * @param k number of clusters.
     */
    public KMeansPlusPlusClusterer(float[][] points, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.length) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
        }

        this.points = points;
        this.k = k;
        random = new Random();
        centroidDenoms = new int[k];
        centroidNums = new float[k][points[0].length];
        centroids = chooseInitialCentroids(points);
        assignments = new int[points.length];

        initializeAssignedPoints();
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @return an array of cluster centroids.
     */
    public float[][] cluster(int maxIterations) {
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
     * @param points a list of points from which centroids are chosen.
     * @return an array of initial centroids.
     */
    private float[][] chooseInitialCentroids(float[][] points) {
        float[][] centroids = new float[k][];
        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        float[] firstCentroid = points[random.nextInt(points.length)];
        centroids[0] = firstCentroid;
        for (int i = 0; i < points.length; i++) {
            float distance1 = VectorUtil.squareDistance(points[i], firstCentroid);
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

            float[] nextCentroid = points[selectedIdx];
            centroids[i] = nextCentroid;

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.length; j++) {
                float newDistance = VectorUtil.squareDistance(points[j], nextCentroid);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }

        for (float[] centroid : centroids) {
            assertFinite(centroid);
        }
        return centroids;
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`.
     * This method should only be called once after initial centroids are chosen.
     */
    private void initializeAssignedPoints() {
        for (int i = 0; i < points.length; i++) {
            float[] point = points[i];
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
            float[] point = points[i];
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
    private int getNearestCluster(float[] point) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < k; i++) {
            float distance = VectorUtil.squareDistance(point, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    @SuppressWarnings({"AssertWithSideEffects", "ConstantConditions"})
    private static void assertFinite(float[] vector) {
        boolean assertsEnabled = false;
        assert assertsEnabled = true;

        if (assertsEnabled) {
            for (float v : vector) {
                assert Float.isFinite(v) : "vector " + Arrays.toString(vector) + " contains non-finite value";
            }
        }
    }

    /**
     * Calculates centroids from centroidNums/centroidDenoms updated during point assignment
     */
    private void updateCentroids() {
        for (int i = 0; i < centroids.length; i++) {
            var denom = centroidDenoms[i];
            if (denom == 0) {
                centroids[i] = points[random.nextInt(points.length)];
            } else {
                centroids[i] = Arrays.copyOf(centroidNums[i], centroidNums[i].length);
                VectorUtil.scale(centroids[i], 1.0f / centroidDenoms[i]);
            }
        }
    }

    /**
     * Computes the centroid of a list of points.
     */
    public static float[] centroidOf(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = VectorUtil.sum(points);
        VectorUtil.scale(centroid, 1.0f / points.size());

        return centroid;
    }

    public float[][] getCentroids() {
        return centroids;
    }
}
