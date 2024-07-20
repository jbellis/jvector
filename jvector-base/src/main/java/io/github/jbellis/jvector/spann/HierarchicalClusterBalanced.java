package io.github.jbellis.jvector.spann;

import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.agrona.collections.IntArrayList;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.Math.max;

// Make this the source of truth for postings lists (add methods to mutate)
//      How does C++ add closure assignments?

/**
 * Implements a Hierarchical Cluster-Balanced (HCB) algorithm for centroid selection.
 */
public class HierarchicalClusterBalanced {
    private static final int MAX_BRANCHING_FACTOR = 32;
    private static final int K_MEANS_ITERATIONS = ProductQuantization.K_MEANS_ITERATIONS;
    private final float INITIAL_LAMBDA = 1.0f;

    private final RandomAccessVectorValues ravv;
    private final VectorSimilarityFunction vsf;

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
     * @param vsf The VectorSimilarityFunction used to compute distances between vectors
     */
    public HierarchicalClusterBalanced(RandomAccessVectorValues ravv, VectorSimilarityFunction vsf)
    {
        this.ravv = ravv;
        this.vsf = vsf;
    }

    /**
     * Selects centroids using the Hierarchical Cluster-Balanced algorithm.
     *
     * @param nCentroids The number of centroids to select
     * @return IdentityHashMap of centroids to ConcurrentSet of assigned point indexes
     * (intended for further updates by caller)
     */
    public List<VectorFloat<?>> computeInitialAssignments(int nCentroids) {
        var allIndices = IntStream.range(0, ravv.size()).collect(IntArrayList::new, IntArrayList::add, IntArrayList::addAll);
        HCBNode root = buildHCBTree(allIndices, max(1, ravv.size() / nCentroids));
        return root.flatten();
    }

    /**
     * Recursively builds the Hierarchical Cluster-Balanced tree.
     */
    public HCBNode buildHCBTree(IntArrayList pointIndices, int idealPointsPerCentroid) {
        if (pointIndices.size() <= idealPointsPerCentroid * 1.5) {
            var centroid = KMeansPlusPlusClusterer.centroidOf(pointIndices.stream().map(ravv::getVector).collect(Collectors.toList()));
            return new HCBNode(centroid, pointIndices);
        }

        return createClusteredTree(ravv, pointIndices, idealPointsPerCentroid, INITIAL_LAMBDA);
    }

    public static HCBNode createClusteredTree(RandomAccessVectorValues ravv, IntArrayList pointIndices, int idealPointsPerCentroid, float lambda) {
        int k = Math.min(MAX_BRANCHING_FACTOR, pointIndices.size() / idealPointsPerCentroid);
        var clusterer = new KMeansPlusPlusBalancedClusterer(pointIndices.toIntArray(), ravv, k, lambda);
        clusterer.cluster(K_MEANS_ITERATIONS);

        var children = new ArrayList<HCBNode>();
        clusterer.getClusters().forEach((cluster, points) -> {
            children.add(buildHCBTree(points, idealPointsPerCentroid));
        });
        return new HCBNode(children);
    }
}
