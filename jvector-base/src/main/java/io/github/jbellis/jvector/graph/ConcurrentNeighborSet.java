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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.RamUsageEstimator;

import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntFunction;

import static java.lang.Math.min;

/** A concurrent set of neighbors that encapsulates diversity/pruning mechanics. */
public class ConcurrentNeighborSet {
    /** the node id whose neighbors we are storing */
    private final int nodeId;

    /**
     * We use a copy-on-write {@link NodeArray} to store the neighbors. Even though updating this is
     * expensive, it is still faster than using a concurrent Collection because "iterate through a
     * node's neighbors" is a hot loop in adding to the graph, and {@link NodeArray} can do that much
     * faster: no boxing/unboxing, all the data is stored sequentially instead of having to follow
     * references, and no fancy encoding necessary for node/score.
     * <p>
     * While GraphIndexBuilder may use approximate scoring to find candidate neighbors, we
     * always rerank them using exact scoring before storing them in the neighbor set.
     */
    private final AtomicReference<Neighbors> neighborsRef;

    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    private final float alpha;

    /** used to compute diversity */
    private final BuildScoreProvider scoreProvider;

    /** the maximum number of neighbors we will end up with */
    private final int maxDegree;
    /** the maximum number of neighbors we can have temporarily during construction */
    private final int maxOverflowDegree;

    /** the proportion of edges that are diverse at alpha=1.0.  updated by retainDiverseInternal */
    private float shortEdges = Float.NaN;

    @VisibleForTesting
    public ConcurrentNeighborSet(int nodeId, int maxDegree, BuildScoreProvider scoreProvider) {
        this(nodeId, maxDegree, maxDegree, scoreProvider, 1.0f);
    }

    public ConcurrentNeighborSet(int nodeId, int maxDegree, int maxOverflowDegree, BuildScoreProvider scoreProvider, float alpha) {
        this(nodeId, maxDegree, maxOverflowDegree, scoreProvider, alpha, new NodeArray(maxOverflowDegree + 1));
    }

    ConcurrentNeighborSet(int nodeId,
                          int maxDegree,
                          int maxOverflowDegree,
                          BuildScoreProvider scoreProvider,
                          float alpha,
                          NodeArray nodes)
    {
        this.nodeId = nodeId;
        this.maxDegree = maxDegree;
        this.maxOverflowDegree = maxOverflowDegree;
        this.scoreProvider = scoreProvider;
        this.alpha = alpha;
        this.neighborsRef = new AtomicReference<>(new Neighbors(nodes, 0));
    }

    public float getShortEdges() {
        return shortEdges;
    }

    public NodesIterator iterator() {
        return new NeighborIterator(neighborsRef.get().nodes);
    }

    /**
     * For every neighbor X that this node Y connects to, add a reciprocal link from X to Y.
     * If overflow is > 1.0, allow the number of neighbors to exceed maxConnections temporarily.
     */
    public void backlink(IntFunction<ConcurrentNeighborSet> neighborhoodOf, float overflow) {
        NodeArray neighbors = neighborsRef.get().nodes;
        for (int i = 0; i < neighbors.size(); i++) {
            int nbr = neighbors.node[i];
            float nbrScore = neighbors.score[i];
            ConcurrentNeighborSet nbrNbr = neighborhoodOf.apply(nbr);
            assert nbrNbr != null : "Node " + nbr + " not found";
            nbrNbr.insert(nodeId, nbrScore, overflow);
        }
    }

    /**
     * Enforce maxConnections as a hard cap, since we allow it to be exceeded temporarily during construction
     * for efficiency.  This method is threadsafe, but if you call it concurrently with other inserts,
     * the limit may end up being exceeded again.
     */
    public void enforceDegree() {
        neighborsRef.getAndUpdate(old -> {
            if (old.nodes.size <= maxDegree) {
                return old;
            }
            var nextNodes = old.nodes.copy();
            retainDiverse(nextNodes, old.diverseBefore, true);
            return new Neighbors(nextNodes, nextNodes.size);
        });
    }

    public void replaceDeletedNeighbors(Bits deletedNodes, NodeArray candidates) {
        neighborsRef.getAndUpdate(old -> {
            // copy the non-deleted neighbors to a new NodeArray
            var liveNeighbors = new NodeArray(old.nodes.size);
            for (int i = 0; i < old.nodes.size(); i++) {
                int node = old.nodes.node[i];
                if (!deletedNodes.get(node)) {
                    liveNeighbors.addInOrder(node, old.nodes.score[i]);
                }
            }

            // merge the remaining neighbors with the candidates
            NodeArray merged = rescoreAndRetainDiverse(liveNeighbors, candidates);
            return new Neighbors(merged, merged.size);
        });
    }

    private static class NeighborIterator extends NodesIterator {
        private final NodeArray neighbors;
        private int i;

        private NeighborIterator(NodeArray neighbors) {
            super(neighbors.size());
            this.neighbors = neighbors;
            i = 0;
        }

        @Override
        public boolean hasNext() {
            return i < neighbors.size();
        }

        @Override
        public int nextInt() {
            return neighbors.node[i++];
        }
    }

    public int size() {
        return neighborsRef.get().nodes.size();
    }

    /**
     * For each candidate (going from best to worst), select it only if it is closer to target than it
     * is to any of the already-selected candidates. This is maintained whether those other neighbors
     * were selected by this method, or were added as a "backlink" to a node inserted concurrently
     * that chose this one as a neighbor.
     */
    public void insertDiverse(NodeArray toMerge) {
        if (toMerge.size() == 0) {
            return;
        }

        neighborsRef.getAndUpdate(old -> {
            // merge all the candidates into a single array and compute the diverse ones to keep
            // from that.  we do this first by selecting the ones to keep, and then by copying
            // only those into a new NeighborArray.  This is less expensive than doing the
            // diversity computation in-place, since we are going to do multiple passes and
            // pruning back extras is expensive.
            NodeArray merged;
            if (old.nodes.size > 0) {
                merged = rescoreAndRetainDiverse(old.nodes, toMerge);
            } else {
                merged = toMerge.copy(); // still need to copy in case we lose the race
                retainDiverse(merged, 0, scoreProvider.isExact());
            }
            // insertDiverse usually gets called with a LOT of candidates, so trim down the resulting NodeArray
            var nextNodes = merged.node.length <= nodeArrayLength() ? merged : merged.copy(nodeArrayLength());
            return new Neighbors(nextNodes, nextNodes.size);
        });
    }

    // does not modify `old` or `toMerge`
    private NodeArray rescoreAndRetainDiverse(NodeArray old, NodeArray toMerge) {
        NodeArray merged;
        if (scoreProvider.isExact()) {
            merged = NodeArray.merge(old, toMerge);
        } else {
            // merge assumes that node X will always have the same score in both arrays, so we need
            // to compute approximate scores for the existing nodes to make the comparison valid.
            // (we expect to have many more new candidates than existing neighbors)
            var approximatedOld = computeApproximatelyScored(old);
            merged = NodeArray.merge(approximatedOld, toMerge);
        }
        // retainDiverse will switch back to exact-scored
        retainDiverse(merged, 0, scoreProvider.isExact());
        return merged;
    }

    private int nodeArrayLength() {
        // one extra so that insert() against a full NodeArray doesn't invoke growArrays()
        return maxOverflowDegree + 1;
    }

    private NodeArray computeApproximatelyScored(NodeArray exact) {
        var approximated = new NodeArray(exact.size);
        var sf = scoreProvider.diversityProvider().createFor(nodeId).scoreFunction();
        assert !sf.isExact();
        for (int i = 0; i < exact.size; i++) {
            approximated.insertSorted(exact.node[i], sf.similarityTo(exact.node[i]));
        }
        return approximated;
    }

    void insertNotDiverse(int node, float score) {
        neighborsRef.getAndUpdate(old -> {
            assert old.nodes.size <= maxDegree : "insertNotDiverse called before enforcing degree/diversity";
            NodeArray nextNodes = old.nodes.copy(maxDegree);
            // remove the worst edge to make room for the new one, if necessary
            nextNodes.size = min(nextNodes.size, maxDegree - 1);
            int insertedAt = nextNodes.insertSorted(node, score);
            if (insertedAt == -1) {
                return old;
            }
            return new Neighbors(nextNodes, min(insertedAt, old.diverseBefore));
        });
    }

    /**
     * Retain the diverse neighbors, updating `neighbors` in place
     */
    private void retainDiverse(NodeArray neighbors, int diverseBefore, boolean isExactScored) {
        BitSet selected = new FixedBitSet(neighbors.size());
        for (int i = 0; i < min(diverseBefore, maxDegree); i++) {
            selected.set(i);
        }

        var dp = scoreProvider.diversityProvider();
        if (isExactScored) {
            // either the provider is natively exact, or we're on the backlink->insert path,
            // so `neighbors` is exact-scored
            retainDiverseInternal(neighbors, maxDegree, diverseBefore, selected, node1 -> dp.createFor(node1).exactScoreFunction());
            neighbors.retain(selected);
        } else {
            // provider is natively approximate and we're on the insertDiverse path
            assert !scoreProvider.isExact();
            assert diverseBefore == 0;

            // rerank with exact scores
            // Note: this is actually faster than computing diversity using approximate scores (and then rescoring only
            // the remaining neighbors), we lose more from loading from dozens of codebook locations in memory than we
            // do from touching disk some extra times to get the exact score
            var sf = dp.createFor(nodeId).exactScoreFunction();
            var exactScoredNeighbors = new NodeArray(neighbors.size);
            for (int i = 0; i < neighbors.size; i++) {
                int neighborId = neighbors.node[i];
                float score = sf.similarityTo(neighborId);
                exactScoredNeighbors.insertSorted(neighborId, score);
            }

            // compute diversity against the exact-scored and reordered list
            retainDiverseInternal(exactScoredNeighbors, maxDegree, 0, selected, node1 -> dp.createFor(node1).exactScoreFunction());

            // copy the final result into the original container
            neighbors.clear();
            for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
                neighbors.addInOrder(exactScoredNeighbors.node[i], exactScoredNeighbors.score[i]);
            }
        }
    }

    private void retainDiverseInternal(NodeArray neighbors, int max, int diverseBefore, BitSet selected, ScoreFunction.Provider scoreProvider) {
        int nSelected = diverseBefore;
        // add diverse candidates, gradually increasing alpha to the threshold
        // (so that the nearest candidates are prioritized)
        for (float a = 1.0f; a <= alpha + 1E-6 && nSelected < max; a += 0.2f) {
            for (int i = diverseBefore; i < neighbors.size() && nSelected < max; i++) {
                if (selected.get(i)) {
                    continue;
                }

                int cNode = neighbors.node()[i];
                float cScore = neighbors.score()[i];
                var sf = scoreProvider.scoreFunctionFor(cNode);
                if (isDiverse(cNode, cScore, neighbors, sf, selected, a)) {
                    selected.set(i);
                    nSelected++;
                }
            }

            if (a == 1.0f && max == maxDegree) {
                // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                // when we don't have to worry about concurrent changes
                shortEdges = nSelected / (float) maxDegree;
            }
        }
    }

    // is the candidate node with the given score closer to the base node than it is to any of the
    // already-selected neighbors
    private boolean isDiverse(int node, float score, NodeArray others, ScoreFunction sf, BitSet selected, float alpha) {
        assert others.size > 0;

        for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
            int otherNode = others.node()[i];
            if (node == otherNode) {
                break;
            }
            if (sf.similarityTo(otherNode) > score * alpha) {
                return false;
            }
        }
        return true;
    }

    NodeArray getCurrent() {
        return neighborsRef.get().nodes;
    }

    /**
     * Insert a new neighbor, maintaining our size cap by removing the least diverse neighbor if
     * necessary. "Overflow" is the factor by which to allow going over the size cap temporarily.
     */
    public void insert(int neighborId, float score, float overflow) {
        assert neighborId != nodeId : "can't add self as neighbor at node " + nodeId;

        int hardMax = (int) (overflow * maxDegree);
        assert hardMax <= maxOverflowDegree : String.format("overflow %s could exceed max overflow degree %d", overflow, maxOverflowDegree);

        neighborsRef.getAndUpdate(old -> {
            NodeArray nextNodes = old.nodes.copy(nodeArrayLength());
            int insertionPoint = nextNodes.insertSorted(neighborId, score);
            if (insertionPoint == -1) {
                return old;
            }

            // batch up the enforcement of the max connection limit, since otherwise
            // we do a lot of duplicate work scanning nodes that we won't remove
            int nextDiverseBefore = min(insertionPoint, old.diverseBefore);
            if (nextNodes.size > hardMax) {
                retainDiverse(nextNodes, nextDiverseBefore, true);
                nextDiverseBefore = nextNodes.size;
            }

            return new Neighbors(nextNodes, nextDiverseBefore);
        });
    }

    public static long ramBytesUsed(int nodes) {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;
        int AREF_BYTES = OH_BYTES + 2 * REF_BYTES;

        return OH_BYTES
                + REF_BYTES + AREF_BYTES // Neighbors AtomicReference
                + OH_BYTES + REF_BYTES + Integer.BYTES // Neighbors
                + NodeArray.ramBytesUsed(nodes) // NodeArray
                + Float.BYTES // alpha
                + REF_BYTES // BSP
                + Integer.BYTES // maxDegree
                + Integer.BYTES // maxOverflowDegree
                + Float.BYTES; // shortEdges
    }

    /** Only for testing; this is a linear search */
    boolean contains(int i) {
        var it = this.iterator();
        while (it.hasNext()) {
            if (it.nextInt() == i) {
                return true;
            }
        }
        return false;
    }

    private static class Neighbors {
        /**
         * The neighbors of the node
         */
        public final NodeArray nodes;

        /**
         * Neighbors up to (but not including) this index are known to be diverse
         */
        public final int diverseBefore;

        private Neighbors(NodeArray nodes, int diverseBefore) {
            this.nodes = nodes;
            this.diverseBefore = diverseBefore;
        }
    }
}
