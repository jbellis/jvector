package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private static final int MAX_BRANCHING_FACTOR = 32;
    private static final int K_MEANS_ITERATIONS = ProductQuantization.K_MEANS_ITERATIONS;
    private final float INITIAL_LAMBDA = 1.0f;

    private final RandomAccessVectorValues ravv;
    private final IntArrayList points;

    /**
     * Represents a node in the Hierarchical Cluster-Balanced tree.
     */
    public static class HCBNode {
        private final VectorFloat<?> centroid;
        private final List<HCBNode> children;
        private final IntArrayList pointIndexes;

        private HCBNode(List<HCBNode> children) {
            this.children = children;
            this.centroid = null;
            this.pointIndexes = null;
        }

        private HCBNode(VectorFloat<?> centroid, IntArrayList pointIndexes) {
            this.centroid = centroid;
            this.pointIndexes = pointIndexes;
            this.children = null;
        }

        public List<VectorFloat<?>> flatten() {
            var centroids = new ArrayList<VectorFloat<?>>();
            this.flatten(centroids);
            return centroids;
        }

        private void flatten(ArrayList<VectorFloat<?>> centroids) {
            if (children == null) {
                assert pointIndexes != null;
                centroids.add(centroid);
                return;
            }

            for (HCBNode child : children) {
                child.flatten(centroids);
            }
        }
    }

    /**
     * Constructs a new HierarchicalClusterBalanced instance.
     *
     * @param ravv The RandomAccessVectorValues containing the vectors to cluster
     */
    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv)
    {
        this(ravv, IntStream.range(0, ravv.size()).collect(IntArrayList::new, IntArrayList::add, IntArrayList::addAll));
    }

    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv, IntArrayList points) {
        this.ravv = ravv;
        this.points = points;
    }

    /**
     * Selects centroids using the Hierarchical Cluster-Balanced algorithm.
     */
    public List<VectorFloat<?>> computeCentroids(int idealAssignments) {
        HCBNode root = buildHCBTree(points, idealAssignments, 0);
        return root.flatten();
    }

    /**
     * Recursively builds the Hierarchical Cluster-Balanced tree.
     */
    private HCBNode buildHCBTree(IntArrayList pointIndices, int idealPointsPerCentroid, int depth) {
        if (depth > 100) {
            throw new IllegalStateException("Too deep");
        }
        if (pointIndices.size() <= idealPointsPerCentroid * 2) {
            var centroid = KMeansPlusPlusClusterer.centroidOf(pointIndices.stream().map(ravv::getVector).collect(Collectors.toList()));
            return new HCBNode(centroid, pointIndices);
        }

        int k = min(MAX_BRANCHING_FACTOR, max(2, pointIndices.size() / idealPointsPerCentroid));
        var clusterer = new KMeansPlusPlusPointsClusterer(pointIndices.toIntArray(), ravv, k);
        clusterer.cluster(K_MEANS_ITERATIONS);

        var children = new ArrayList<HCBNode>();
        clusterer.getClusters().forEach((cluster, clusterPoints) -> {
            children.add(buildHCBTree(clusterPoints, idealPointsPerCentroid, depth + 1));
        });
        return new HCBNode(children);
    }

}
