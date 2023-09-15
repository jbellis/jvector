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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

/**
 * A KMeans++ implementation for float vectors.  Optimizes to use SIMD vector
 * instructions, and to use the triangle inequality to skip distance calculations.
 * Roughly 3x faster than using the apache commons math implementation (with
 * conversions to double[]).
 */
public class KMeansPlusPlusClusterer {
    private final int k;
    private final BiFunction<float[], float[], Float> distanceFunction;
    private final Random random;
    private final List<float[]>[] clusterPoints;
    private final float[][] centroidDistances;
    private final float[][] points;
    private final int[] assignments;
    private final float[][] centroids;


    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified number of clusters,
     * maximum iterations, and distance function.
     *
     * @param k number of clusters.
     * @param distanceFunction a function to compute the distance between two points.
     */
    public KMeansPlusPlusClusterer(float[][] points, int k, BiFunction<float[], float[], Float> distanceFunction) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.length) {
            throw new IllegalArgumentException(String.format("Number of clusters %d cannot exceed number of points %d", k, points.length));
        }

        this.points = points;
        this.k = k;
        this.distanceFunction = distanceFunction;
        this.random = new Random();
        this.clusterPoints = new List[k];
        for (int i = 0; i < k; i++) {
            this.clusterPoints[i] = new ArrayList<>();
        }
        centroidDistances = new float[k][k];
        centroids = chooseInitialCentroids(points);
        updateCentroidDistances();
        assignments = new int[points.length];
        assignPointsToClusters();
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @return a list of cluster centroids.
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
        for (int j = 0; j < k; j++) {
            if (clusterPoints[j].isEmpty()) {
                // Handle empty cluster by choosing a random point
                // (Choosing the highest-variance point is much slower and no better after a couple iterations)
                centroids[j] = points[random.nextInt(points.length)];
            } else {
                centroids[j] = centroidOf(clusterPoints[j]);
            }
        }
        int changedCount = assignPointsToClusters();
        updateCentroidDistances();

        return changedCount;
    }

    private void updateCentroidDistances() {
        for (int m = 0; m < k; m++) {
            for (int n = m + 1; n < k; n++) {
                float distance = distanceFunction.apply(centroids[m], centroids[n]);
                centroidDistances[m][n] = distance;
                centroidDistances[n][m] = distance; // Distance matrix is symmetric
            }
        }
    }

    /**
     * Chooses the initial centroids for clustering.
     *
     * The first centroid is chosen randomly from the data points. Subsequent centroids
     * are selected with a probability proportional to the square of their distance
     * to the nearest existing centroid. This ensures that the centroids are spread out
     * across the data and not initialized too closely to each other, leading to better
     * convergence and potentially improved final clusterings.
     * *
     * @param points a list of points from which centroids are chosen.
     * @return a list of initial centroids.
     */
    private float[][] chooseInitialCentroids(float[][] points) {
        float[][] centroids = new float[k][];
        float[] distances = new float[points.length];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        float[] firstCentroid = points[random.nextInt(points.length)];
        centroids[0] = firstCentroid;
        for (int i = 0; i < points.length; i++) {
            float distance1 = distanceFunction.apply(points[i], firstCentroid);
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
                float newDistance = distanceFunction.apply(points[j], nextCentroid);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }

        return centroids;
    }

    /**
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`
     */
    private int assignPointsToClusters() {
        int changedCount = 0;

        for (List<float[]> cluster : clusterPoints) {
            cluster.clear();
        }

        for (int i = 0; i < points.length; i++) {
            float[] point = points[i];
            int clusterIndex = getNearestCluster(point, centroids);

            // Check if assignment has changed
            if (assignments[i] != clusterIndex) {
                changedCount++;
            }

            clusterPoints[clusterIndex].add(point);
            assignments[i] = clusterIndex;
        }

        return changedCount;
    }

    /**
     * Return the index of the closest centroid to the given point
     */
    private int getNearestCluster(float[] point, float[][] centroids) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < k; i++) {
            if (i != nearestCluster) {
                // Using triangle inequality to potentially skip the computation
                float potentialMinDistance = Math.abs(minDistance - centroidDistances[nearestCluster][i]);
                if (potentialMinDistance >= minDistance) {
                    continue;
                }
            }

            float distance = distanceFunction.apply(point, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    /**
     * Computes the centroid of a list of points.
     */
    public static float[] centroidOf(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = VectorUtil.sum(points);
        VectorUtil.divInPlace(centroid, points.size());

        return centroid;
    }
}
