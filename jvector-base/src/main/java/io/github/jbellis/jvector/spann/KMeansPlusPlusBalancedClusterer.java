package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntArrayList;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.vector.VectorUtil.addInPlace;
import static io.github.jbellis.jvector.vector.VectorUtil.scale;

public class KMeansPlusPlusBalancedClusterer {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    // Maximum number of assignments for each vector
    private static final int MAX_ASSIGNMENTS = 8;

    // number of centroids to compute
    private final int k;

    // the points to train on
    private final int[] points;
    // source of vectors corresponding to points
    private final RandomAccessVectorValues ravv;
    // the clusters each point is assigned to
    private final IntArrayList[] assignments;
    // the centroids of each cluster
    private final VectorFloat<?>[] centroids;

    // used to accelerate updating clusters
    private final float[] centroidDenoms; // the fractional number of points assigned to each cluster
    private final VectorFloat<?>[] centroidNums; // the weighted sum of all points assigned to each cluster

    // comparison function
    private final VectorSimilarityFunction vsf;

    // Lambda parameter for balanced clustering
    private final float lambda;
    private int closureThreshold;

    /**
     * Constructs a KMeansPlusPlusBalancedClusterer with the specified points and initial centroids.
     */
    public KMeansPlusPlusBalancedClusterer(int[] points, RandomAccessVectorValues ravv, int k, VectorSimilarityFunction vsf, float lambda) {
        this.ravv = ravv;
        this.points = points;
        this.k = k;
        this.centroids = chooseInitialCentroids(points, ravv, k, vsf);
        this.vsf = vsf;
        this.lambda = lambda;

        centroidDenoms = new float[k];
        centroidNums = new VectorFloat<?>[k];
        for (int i = 0; i < k; i++) {
            centroidNums[i] = vectorTypeSupport.createFloatVector(ravv.dimension());
        }
        assignments = new IntArrayList[k];
        for (int i = 0; i < k; i++) {
            assignments[i] = new IntArrayList();
        }

        closureThreshold = computeClosureThreshold();
        initializeAssignedPoints();
    }

    /** compute a threshold such that on average each point will have MAX_ASSIGNMENTS/2 assignments */
    private int computeClosureThreshold() {
        // nearest is broken, needs to sort
        // should we dynamically adjust threshold as we compute or do it once up front for the entire HCB?
    }

    /**
     * Refines the lambda value based on the current clustering results.
     */
    public float getRefinedLambda() {
        int maxCluster = IntStream.range(0, k).reduce((a, b) -> centroidDenoms[a] > centroidDenoms[b] ? a : b).orElse(-1);
        if (maxCluster == -1) {
            return lambda;
        }

        VectorFloat<?> maxCenter = centroids[maxCluster];
        DoubleAdder totalDist = new DoubleAdder();
        AtomicInteger count = new AtomicInteger();

        IntStream.range(0, points.length).parallel().forEach(i -> {
            VectorFloat<?> point = ravv.getVector(points[i]);
            if (getNearestCluster(point) == maxCluster) {
                totalDist.add(vsf.compare(maxCenter, point));
                count.incrementAndGet();
            }
        });
        double avgDist = totalDist.doubleValue() / count.get();

        // Adjust lambda based on the average distance in the largest cluster
        return (float) ((getMaxClusterDist(maxCluster) - avgDist) / points.length);
    }

    /**
     * Performs clustering on the provided set of points.
     */
    public void cluster(int iterations) {
        for (int i = 0; i < iterations; i++) {
            updateCentroids();
            int changedCount = updateAssignedPoints();
            if (changedCount <= 0.01 * points.length) {
                System.out.println("Converged after " + i + " iterations");
                break;
            }
        }
    }

    /**
     * Chooses the initial centroids for clustering.
     */
    private static VectorFloat<?>[] chooseInitialCentroids(int[] points, RandomAccessVectorValues ravv, int k, VectorSimilarityFunction vsf) {
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
            float distance1 = vsf.compare(ravv.getVector(points[i]), centroids[0]);
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

            centroids[i] = ravv.getVector(points[selectedIdx]).copy();

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.length; j++) {
                float newDistance = vsf.compare(ravv.getVector(points[j]), centroids[i]);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }
        return centroids;
    }

    /**
     * Assigns points to the nearest clusters. This method should only be called once after initial centroids are chosen.
     */
    private void initializeAssignedPoints() {
        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = ravv.getVector(points[i]);
            var nearestClusters = getNearestClusters(point);
            float weight = 1.0f / nearestClusters.size();
            for (int cluster : nearestClusters) {
                assignments[cluster].add(i);
                centroidDenoms[cluster] += weight;
                var scaled = point.copy();
                scale(scaled, weight);
                addInPlace(centroidNums[cluster], scaled);
            }
        }
    }

    /**
     * Assigns points to the nearest clusters.
     * @return the number of points that changed clusters
     */
    private int updateAssignedPoints() {
        var oldAssignments = new IntArrayList[k];
        for (int i = 0; i < k; i++) {
            var a = new IntArrayList(assignments[i].size(), Integer.MIN_VALUE);
            a.addAll(assignments[i]);
            oldAssignments[i] = a;
            assignments[i].clear();
        }

        for (int i = 0; i < points.length; i++) {
            VectorFloat<?> point = ravv.getVector(points[i]);
            var newAssignments = getNearestClusters(point);

            float weight = 1.0f / newAssignments.size();
            for (int cluster : newAssignments) {
                assignments[cluster].add(i);
                centroidDenoms[cluster] += weight;
                var scaled = point.copy();
                scale(scaled, weight);
                addInPlace(centroidNums[cluster], scaled);
            }
        }

        return IntStream.range(0, k).map(i -> {
            var old = oldAssignments[i];
            var current = assignments[i];
            return old.equals(current) ? 0 : 1;
        }).sum();
    }

    /**
     * Return the indices of the closest centroids to the given point
     */
    private IntArrayList getNearestClusters(VectorFloat<?> point) {
        float[] distances = new float[k];
        for (int i = 0; i < k; i++) {
            distances[i] = vsf.compare(point, centroids[i]) + lambda * centroidDenoms[i];
        }

        float minDistance = Float.MAX_VALUE;
        for (float distance : distances) {
            if (distance < minDistance) {
                minDistance = distance;
            }
        }

        IntArrayList nearestClusters = new IntArrayList(MAX_ASSIGNMENTS, Integer.MIN_VALUE);
        for (int i = 0; i < k; i++) {
            if (distances[i] <= minDistance * (1 + closureThreshold)) {
                nearestClusters.add(i);
                if (nearestClusters.size() >= MAX_ASSIGNMENTS) {
                    break;
                }
            }
        }

        return nearestClusters;
    }

    /**
     * Return the index of the single closest centroid to the given point
     */
    private int getNearestCluster(VectorFloat<?> point) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < k; i++) {
            float distance = vsf.compare(point, centroids[i]) + lambda * centroidDenoms[i];
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
            if (centroidDenoms[i] == 0) {
                // no points assigned to this cluster
                initializeCentroidToRandomPoint(i);
            } else {
                var centroid = centroidNums[i].copy();
                scale(centroid, 1.0f / centroidDenoms[i]);
                centroids[i] = centroid;
            }
            // Reset for next iteration
            centroidDenoms[i] = 0;
            centroidNums[i] = vectorTypeSupport.createFloatVector(ravv.dimension());
        }
    }

    private void initializeCentroidToRandomPoint(int i) {
        var random = ThreadLocalRandom.current();
        centroids[i] = ravv.getVector(points[random.nextInt(points.length)]).copy();
    }

    public Map<VectorFloat<?>, IntArrayList> getClusters() {
        var clusters = new IdentityHashMap<VectorFloat<?>, IntArrayList>();
        for (int i = 0; i < k; i++) {
            clusters.put(centroids[i], assignments[i]);
        }
        return clusters;
    }

    /**
     * @return the largest distance between any point and the centroid, for the given cluster
     */
    private float getMaxClusterDist(int cluster) {
        float maxDist = 0;
        for (int i = 0; i < assignments[cluster].size(); i++) {
            int pointIndex = assignments[cluster].get(i);
            float dist = vsf.compare(ravv.getVector(points[pointIndex]), centroids[cluster]);
            maxDist = Math.max(maxDist, dist);
        }
        return maxDist;
    }
}

