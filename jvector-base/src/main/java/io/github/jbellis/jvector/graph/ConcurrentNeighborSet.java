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

import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import org.agrona.collections.IntHashSet;

import java.util.concurrent.atomic.AtomicBoolean;
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
     */
    private final AtomicReference<Neighbors> neighborsRef;

    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    private final float alpha;

    /** used to compute diversity */
    private final BuildScoreProvider scoreProvider;

    /** the maximum number of neighbors we can store */
    private final int maxConnections;

    /** the proportion of edges that are diverse at alpha=1.0.  updated by removeAllNonDiverse */
    private float shortEdges = Float.NaN;

    public ConcurrentNeighborSet(int nodeId, int maxConnections, BuildScoreProvider scoreProvider) {
        this(nodeId, maxConnections, scoreProvider, 1.0f);
    }

    public ConcurrentNeighborSet(int nodeId, int maxConnections, BuildScoreProvider scoreProvider, float alpha) {
        this(nodeId, maxConnections, scoreProvider, alpha, new NodeArray(maxConnections));
    }

    ConcurrentNeighborSet(int nodeId,
                          int maxConnections,
                          BuildScoreProvider scoreProvider,
                          float alpha,
                          NodeArray nodes)
    {
        this.nodeId = nodeId;
        this.maxConnections = maxConnections;
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
            var nodes = removeAllNonDiverse(old.nodes, old.diverseBefore);
            return new Neighbors(nodes, nodes.size);
        });
    }

    /**
     * @return true if we had deleted neighbors
     */
    public boolean removeDeletedNeighbors(Bits deletedNodes) {
        AtomicBoolean found = new AtomicBoolean();
        neighborsRef.getAndUpdate(old -> {
            var nodes = old.nodes;
            int nextDiverseBefore = old.diverseBefore;
            // build a set of the entries we want to retain
            var toRetain = new FixedBitSet(nodes.size);
            for (int i = 0; i < nodes.size; i++) {
                if (deletedNodes.get(nodes.node[i])) {
                    found.set(true);
                    if (i < nextDiverseBefore) {
                        nextDiverseBefore--;
                    }
                } else {
                    toRetain.set(i);
                }
            }

            // if we're retaining everything, no need to make a copy
            if (!found.get()) {
                return old;
            }

            // copy and purge the deleted ones
            var nextNodes = nodes.copy();
            nextNodes.retain(toRetain);
            return new Neighbors(nextNodes, nextDiverseBefore);
        });
        return found.get();
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
                if (!scoreProvider.isExact()) {
                    // merge assumes that node X will always have the same score in both arrays, so we need
                    // to compute approximate scores for the existing nodes to make the comparison valid
                    var approximatedOld = new NodeArray(old.nodes.size);
                    var sf = scoreProvider.scoreFunctionFor(nodeId);
                    for (int i = 0; i < old.nodes.size; i++) {
                        approximatedOld.insertSorted(old.nodes.node[i], sf.similarityTo(old.nodes.node[i]));
                    }
                    merged = mergeNeighbors(approximatedOld, toMerge);
                } else {
                    merged = mergeNeighbors(old.nodes, toMerge);
                }
            } else {
                merged = toMerge.copy(); // still need to copy in case we lose the race
            }
            retainDiverse(merged, 0, scoreProvider.isExact());
            return new Neighbors(merged, merged.size);
        });
    }

    void padWith(NodeArray connections) {
        // we deliberately do not perform diversity checks here
        // (it will be invoked when the cleanup code calls insertDiverse later
        // with the results of the nn descent rebuild)
        neighborsRef.getAndUpdate(old -> new Neighbors(mergeNeighbors(old.nodes, connections), 0));
    }

    void insertNotDiverse(int node, float score) {
        neighborsRef.getAndUpdate(old -> {
            NodeArray nextNodes = old.nodes.copy();
            // remove the worst edge to make room for the new one, if necessary
            nextNodes.size = min(nextNodes.size, maxConnections - 1);
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
        for (int i = 0; i < min(diverseBefore, maxConnections); i++) {
            selected.set(i);
        }

        if (isExactScored) {
            retainDiverseInternal(neighbors, maxConnections, diverseBefore, selected, scoreProvider::scoreFunctionFor);
        } else {
            // reranking for diversity!  first pass against the approximate scores
            assert diverseBefore == 0;
            retainDiverseInternal(neighbors, 2 * maxConnections, 0, selected, scoreProvider::scoreFunctionFor);
            neighbors.retain(selected);

            // compute exact scores
            var exactScores = scoreProvider.rerankerFor(nodeId).score(neighbors.node);
            var exactScoredNeighbors = new NodeArray(neighbors.size);
            for (int i = 0; i < neighbors.size; i++) {
                int node = neighbors.node[i];
                float score = exactScores.get(i);
                exactScoredNeighbors.insertSorted(node, score);
            }

            // second pass
            selected.clear();
            retainDiverseInternal(exactScoredNeighbors, maxConnections, 0, selected, node1 -> scoreProvider.rerankerFor(node1).scoreFunction());

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
                if (isDiverse(cNode, cScore, neighbors, scoreProvider.scoreFunctionFor(cNode), selected, a)) {
                    selected.set(i);
                    nSelected++;
                }
            }

            if (a == 1.0f && max == maxConnections) {
                // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                // when we don't have to worry about concurrent changes
                shortEdges = nSelected / (float) maxConnections;
            }
        }

        neighbors.retain(selected);
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

            // nextSetBit will error out if we're at the end of the bitset, so check this manually
            if (i + 1 >= selected.length()) {
                break;
            }
        }
        return true;
    }

    private NodeArray removeAllNonDiverse(NodeArray neighbors, int diverseBefore) {
        if (neighbors.size <= maxConnections) {
            return neighbors;
        }
        var copy = neighbors.copy();
        retainDiverse(copy, diverseBefore, true);
        return copy;
    }

    NodeArray getCurrent() {
        return neighborsRef.get().nodes;
    }

    // TODO move this to NodeArray
    static NodeArray mergeNeighbors(NodeArray a1, NodeArray a2) {
        NodeArray merged = new NodeArray(a1.size() + a2.size());
        int i = 0, j = 0;

        // since nodes are only guaranteed to be sorted by score -- ties can appear in any node order --
        // we need to remember all the nodes with the current score to avoid adding duplicates
        var nodesWithLastScore = new IntHashSet();
        float lastAddedScore = Float.NaN;

        // loop through both source arrays, adding the highest score element to the merged array,
        // until we reach the end of one of the sources
        while (i < a1.size() && j < a2.size()) {
            if (a1.score()[i] < a2.score[j]) {
                // add from a2
                if (a2.score[j] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a2.score[j];
                }
                if (nodesWithLastScore.add(a2.node[j])) {
                    merged.addInOrder(a2.node[j], a2.score[j]);
                }
                j++;
            } else if (a1.score()[i] > a2.score[j]) {
                // add from a1
                if (a1.score()[i] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a1.score()[i];
                }
                if (nodesWithLastScore.add(a1.node()[i])) {
                    merged.addInOrder(a1.node()[i], a1.score()[i]);
                }
                i++;
            } else {
                // same score -- add both
                if (a1.score()[i] != lastAddedScore) {
                    nodesWithLastScore.clear();
                    lastAddedScore = a1.score()[i];
                }
                if (nodesWithLastScore.add(a1.node()[i])) {
                    merged.addInOrder(a1.node()[i], a1.score()[i]);
                }
                if (nodesWithLastScore.add(a2.node()[j])) {
                    merged.addInOrder(a2.node[j], a2.score[j]);
                }
                i++;
                j++;
            }
        }

        // If elements remain in a1, add them
        if (i < a1.size()) {
            // avoid duplicates while adding nodes with the same score
            while (i < a1.size && a1.score()[i] == lastAddedScore) {
                if (!nodesWithLastScore.contains(a1.node()[i])) {
                    merged.addInOrder(a1.node()[i], a1.score()[i]);
                }
                i++;
            }
            // the remaining nodes have a different score, so we can bulk-add them
            System.arraycopy(a1.node, i, merged.node, merged.size, a1.size - i);
            System.arraycopy(a1.score, i, merged.score, merged.size, a1.size - i);
            merged.size += a1.size - i;
        }

        // If elements remain in a2, add them
        if (j < a2.size()) {
            // avoid duplicates while adding nodes with the same score
            while (j < a2.size && a2.score[j] == lastAddedScore) {
                if (!nodesWithLastScore.contains(a2.node[j])) {
                    merged.addInOrder(a2.node[j], a2.score[j]);
                }
                j++;
            }
            // the remaining nodes have a different score, so we can bulk-add them
            System.arraycopy(a2.node, j, merged.node, merged.size, a2.size - j);
            System.arraycopy(a2.score, j, merged.score, merged.size, a2.size - j);
            merged.size += a2.size - j;
        }

        return merged;
    }

    /**
     * Insert a new neighbor, maintaining our size cap by removing the least diverse neighbor if
     * necessary. "Overflow" is the factor by which to allow going over the size cap temporarily.
     */
    public void insert(int neighborId, float score, float overflow) {
        assert neighborId != nodeId : "can't add self as neighbor at node " + nodeId;
        neighborsRef.getAndUpdate(old -> {
            NodeArray nextNodes = old.nodes.copy();
            int insertionPoint = nextNodes.insertSorted(neighborId, score);
            if (insertionPoint == -1) {
                return old;
            }

            // batch up the enforcement of the max connection limit, since otherwise
            // we do a lot of duplicate work scanning nodes that we won't remove
            int nextDiverseBefore = min(insertionPoint, old.diverseBefore);
            var hardMax = overflow * maxConnections;
            if (nextNodes.size > hardMax) {
                nextNodes = removeAllNonDiverse(nextNodes, nextDiverseBefore);
                nextDiverseBefore = nextNodes.size;
            }

            return new Neighbors(nextNodes, nextDiverseBefore);
        });
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
