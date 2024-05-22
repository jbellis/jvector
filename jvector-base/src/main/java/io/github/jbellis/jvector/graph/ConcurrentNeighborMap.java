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
import io.github.jbellis.jvector.util.DenseIntMap;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.util.RamUsageEstimator;

import static java.lang.Math.min;

/**
 * Encapsulates operations on a graph's neighbors.
 */
public class ConcurrentNeighborMap {
    private final DenseIntMap<Neighbors> neighbors;

    /** the diversity threshold; 1.0 is equivalent to HNSW; Vamana uses 1.2 or more */
    private final float alpha;

    /** used to compute diversity */
    private BuildScoreProvider scoreProvider;

    /** the maximum number of neighbors desired per node */
    public final int maxDegree;
    /** the maximum number of neighbors a node can have temporarily during construction */
    public final int maxOverflowDegree;

    public ConcurrentNeighborMap(BuildScoreProvider scoreProvider, int maxDegree, int maxOverflowDegree, float alpha) {
        this.alpha = alpha;
        this.scoreProvider = scoreProvider;
        this.maxDegree = maxDegree;
        this.maxOverflowDegree = maxOverflowDegree;
        neighbors = new DenseIntMap<>(1024);
    }

    public void insertOne(int fromId, int toId, float score, float overflow) {
        while (true) {
            var old = neighbors.get(fromId);
            var next = old.insert(toId, score, overflow, this);
            if (next == old || neighbors.compareAndPut(fromId, old, next)) {
                break;
            }
        }
    }

    public void insertNotDiverse(int fromId, int toId, float score) {
        while (true) {
            var old = neighbors.get(fromId);
            var next = old.insertNotDiverse(toId, score, this);
            if (next == old || neighbors.compareAndPut(fromId, old, next)) {
                break;
            }
        }
    }

    public void enforceDegree(int nodeId) {
        var old = neighbors.get(nodeId);
        if (old == null) {
            return;
        }

        while (true) {
            old = neighbors.get(nodeId);
            var next = old.enforceDegree(this);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                break;
            }
        }
    }

    public void replaceDeletedNeighbors(int nodeId, BitSet toDelete, NodeArray candidates) {
        while (true) {
            var old = neighbors.get(nodeId);
            var next = old.replaceDeletedNeighbors(toDelete, candidates, this);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                break;
            }
        }
    }

    public Neighbors insertDiverse(int nodeId, NodeArray natural) {
        while (true) {
            var old = neighbors.get(nodeId);
            var next = old.insertDiverse(natural, this);
            if (next == old || neighbors.compareAndPut(nodeId, old, next)) {
                return next;
            }
        }
    }

    public Neighbors get(int node) {
        return neighbors.get(node);
    }

    public int size() {
        return neighbors.size();
    }

    public void addNode(int nodeId, NodeArray nodes) {
        var nodeNeighbors = new Neighbors(nodeId, nodes, 0);
        if (!neighbors.compareAndPut(nodeId, null, nodeNeighbors)) {
            throw new IllegalStateException("Node " + nodeId + " already exists");
        }
    }

    public void addNode(int nodeId) {
        addNode(nodeId, new NodeArray(0));
    }

    public NodesIterator nodesIterator() {
        return neighbors.keysIterator();
    }

    public Neighbors remove(int node) {
        return neighbors.remove(node);
    }

    public boolean contains(int nodeId) {
        return neighbors.containsKey(nodeId);
    }

    public void forEach(DenseIntMap.IntBiConsumer<Neighbors> consumer) {
        neighbors.forEach(consumer);
    }

    public void setScoreProvider(BuildScoreProvider bsp) {
        scoreProvider = bsp;
    }

    private int nodeArrayLength() {
        // one extra so that insert() against a full NodeArray doesn't invoke growArrays()
        return maxOverflowDegree + 1;
    }

    /**
     * Add a link from every node in the NodeArray to the target toId.
     * If overflow is > 1.0, allow the number of neighbors to exceed maxConnections temporarily.
     */
    public void backlink(NodeArray nodes, int toId, float overflow) {
        for (int i = 0; i < nodes.size(); i++) {
            int nbr = nodes.node[i];
            float nbrScore = nodes.score[i];
            insertOne(nbr, toId, nbrScore, overflow);
        }
    }

    /**
     * A concurrent set of neighbors that encapsulates diversity/pruning mechanics.
     * <p>
     * Nothing is modified in place; all mutating methods return a new instance.  These methods
     * are private and should only be exposed through the parent ConcurrentNeighborMap, which
     * performs the appropriate CAS dance.
     * <p>
     * CNM is passed as an explicit parameter to these methods (instead of making this a non-static
     * inner class) to avoid the overhead on the heap of the CNM$this reference.
     */
    public static class Neighbors {
        /** the node id whose neighbors we are storing */
        private final int nodeId;

        /** the proportion of edges that are diverse at alpha=1.0.  updated by retainDiverseInternal */
        private float shortEdges = Float.NaN;

        /** our neighbors and their scores */
        private final NodeArray nodes;

        /** entries in `nodes` before this index are diverse and don't need to be checked again */
        private final int diverseBefore;

        Neighbors(int nodeId, NodeArray nodes, int diverseBefore)
        {
            this.nodeId = nodeId;
            this.diverseBefore = diverseBefore;
            this.nodes = nodes;
        }

        public float getShortEdges() {
            return shortEdges;
        }

        public NodesIterator iterator() {
            return new NeighborIterator(nodes);
        }

        /**
         * Enforce maxConnections as a hard cap, since we allow it to be exceeded temporarily during construction
         * for efficiency.  This method is threadsafe, but if you call it concurrently with other inserts,
         * the limit may end up being exceeded again.
         */
        private Neighbors enforceDegree(ConcurrentNeighborMap map) {
            if (nodes.size <= map.maxDegree) {
                return this;
            }
            var nextNodes = nodes.copy();
            retainDiverse(nextNodes, diverseBefore, true, map);
            return new Neighbors(nodeId, nextNodes, nextNodes.size);
        }

        private Neighbors replaceDeletedNeighbors(Bits deletedNodes, NodeArray candidates, ConcurrentNeighborMap map) {
            // copy the non-deleted neighbors to a new NodeArray
            var liveNeighbors = new NodeArray(nodes.size);
            for (int i = 0; i < nodes.size(); i++) {
                int node = nodes.node[i];
                if (!deletedNodes.get(node)) {
                    liveNeighbors.addInOrder(node, nodes.score[i]);
                }
            }

            // merge the remaining neighbors with the candidates
            NodeArray merged = rescoreAndRetainDiverse(liveNeighbors, candidates, map);
            return new Neighbors(nodeId, merged, merged.size);
        }

        public int size() {
            return nodes.size();
        }

        /**
         * For each candidate (going from best to worst), select it only if it is closer to target than it
         * is to any of the already-selected candidates. This is maintained whether those other neighbors
         * were selected by this method, or were added as a "backlink" to a node inserted concurrently
         * that chose this one as a neighbor.
         */
        private Neighbors insertDiverse(NodeArray toMerge, ConcurrentNeighborMap map) {
            if (toMerge.size() == 0) {
                return this;
            }

            // merge all the candidates into a single array and compute the diverse ones to keep
            // from that.  we do this first by selecting the ones to keep, and then by copying
            // only those into a new NeighborArray.  This is less expensive than doing the
            // diversity computation in-place, since we are going to do multiple passes and
            // pruning back extras is expensive.
            NodeArray merged;
            if (nodes.size > 0) {
                merged = rescoreAndRetainDiverse(nodes, toMerge, map);
            } else {
                merged = toMerge.copy(); // still need to copy in case we lose the race
                retainDiverse(merged, 0, map.scoreProvider.isExact(), map);
            }
            // insertDiverse usually gets called with a LOT of candidates, so trim down the resulting NodeArray
            var nextNodes = merged.node.length <= map.nodeArrayLength() ? merged : merged.copy(map.nodeArrayLength());
            return new Neighbors(nodeId, nextNodes, nextNodes.size);
        }

        // does not modify `old` or `toMerge`
        private NodeArray rescoreAndRetainDiverse(NodeArray old, NodeArray toMerge, ConcurrentNeighborMap map) {
            NodeArray merged;
            if (map.scoreProvider.isExact()) {
                merged = NodeArray.merge(old, toMerge);
            } else {
                // merge assumes that node X will always have the same score in both arrays, so we need
                // to compute approximate scores for the existing nodes to make the comparison valid.
                // (we expect to have many more new candidates than existing neighbors)
                var approximatedOld = computeApproximatelyScored(old, map);
                merged = NodeArray.merge(approximatedOld, toMerge);
            }
            // retainDiverse will switch back to exact-scored
            retainDiverse(merged, 0, map.scoreProvider.isExact(), map);
            return merged;
        }

        private NodeArray computeApproximatelyScored(NodeArray exact, ConcurrentNeighborMap map) {
            var approximated = new NodeArray(exact.size);
            var sf = map.scoreProvider.diversityProvider().createFor(nodeId).scoreFunction();
            assert !sf.isExact();
            for (int i = 0; i < exact.size; i++) {
                approximated.insertSorted(exact.node[i], sf.similarityTo(exact.node[i]));
            }
            return approximated;
        }

        private Neighbors insertNotDiverse(int node, float score, ConcurrentNeighborMap map) {
            int maxDegree = map.maxDegree;
            assert nodes.size <= maxDegree : "insertNotDiverse called before enforcing degree/diversity";
            NodeArray nextNodes = nodes.copy(maxDegree);
            // remove the worst edge to make room for the new one, if necessary
            nextNodes.size = min(nextNodes.size, maxDegree - 1);
            int insertedAt = nextNodes.insertSorted(node, score);
            if (insertedAt == -1) {
                return this;
            }
            return new Neighbors(nodeId, nextNodes, min(insertedAt, diverseBefore));
        }

        /**
         * Retain the diverse neighbors, updating `neighbors` in place
         */
        private void retainDiverse(NodeArray neighbors, int diverseBefore, boolean isExactScored, ConcurrentNeighborMap map) {
            BitSet selected = new FixedBitSet(neighbors.size());
            for (int i = 0; i < min(diverseBefore, map.maxDegree); i++) {
                selected.set(i);
            }

            var dp = map.scoreProvider.diversityProvider();
            if (isExactScored) {
                // either the provider is natively exact, or we're on the backlink->insert path,
                // so `neighbors` is exact-scored
                retainDiverseInternal(neighbors, map.maxDegree, diverseBefore, selected, node1 -> dp.createFor(node1).exactScoreFunction(), map);
                neighbors.retain(selected);
            } else {
                // provider is natively approximate and we're on the insertDiverse path
                assert !map.scoreProvider.isExact();
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
                retainDiverseInternal(exactScoredNeighbors, map.maxDegree, 0, selected, node1 -> dp.createFor(node1).exactScoreFunction(), map);

                // copy the final result into the original container
                neighbors.clear();
                for (int i = selected.nextSetBit(0); i != DocIdSetIterator.NO_MORE_DOCS; i = selected.nextSetBit(i + 1)) {
                    neighbors.addInOrder(exactScoredNeighbors.node[i], exactScoredNeighbors.score[i]);
                }
            }
        }

        private void retainDiverseInternal(NodeArray neighbors, int max, int diverseBefore, BitSet selected, ScoreFunction.Provider scoreProvider, ConcurrentNeighborMap map) {
            int nSelected = diverseBefore;
            // add diverse candidates, gradually increasing alpha to the threshold
            // (so that the nearest candidates are prioritized)
            for (float a = 1.0f; a <= map.alpha + 1E-6 && nSelected < max; a += 0.2f) {
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

                if (a == 1.0f && max == map.maxDegree) {
                    // this isn't threadsafe, but (for now) we only care about the result after calling cleanup(),
                    // when we don't have to worry about concurrent changes
                    shortEdges = nSelected / (float) max;
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
            return nodes;
        }

        /**
         * Insert a new neighbor, maintaining our size cap by removing the least diverse neighbor if
         * necessary. "Overflow" is the factor by which to allow going over the size cap temporarily.
         */
        public Neighbors insert(int neighborId, float score, float overflow, ConcurrentNeighborMap map) {
            assert neighborId != nodeId : "can't add self as neighbor at node " + nodeId;

            int hardMax = (int) (overflow * map.maxDegree);
            assert hardMax <= map.maxOverflowDegree
                    : String.format("overflow %s could exceed max overflow degree %d", overflow, map.maxOverflowDegree);

            NodeArray nextNodes = nodes.copy(map.nodeArrayLength());
            int insertionPoint = nextNodes.insertSorted(neighborId, score);
            if (insertionPoint == -1) {
                return this;
            }

            // batch up the enforcement of the max connection limit, since otherwise
            // we do a lot of duplicate work scanning nodes that we won't remove
            int nextDiverseBefore = min(insertionPoint, diverseBefore);
            if (nextNodes.size > hardMax) {
                retainDiverse(nextNodes, nextDiverseBefore, true, map);
                nextDiverseBefore = nextNodes.size;
            }

            return new Neighbors(nodeId, nextNodes, nextDiverseBefore);
        }

        /** Only for testing; this is a linear search */
        @VisibleForTesting
        boolean contains(int i) {
            var it = this.iterator();
            while (it.hasNext()) {
                if (it.nextInt() == i) {
                    return true;
                }
            }
            return false;
        }
    }

    // Not part of Neighbor because under JDK 11, inner classes may not have static methods
    public static long neighborRamBytesUsed(int count) {
        int REF_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_REF;
        int OH_BYTES = RamUsageEstimator.NUM_BYTES_OBJECT_HEADER;

        return OH_BYTES
                + REF_BYTES + NodeArray.ramBytesUsed(count) // NodeArray
                + Integer.BYTES // nodeId
                + Integer.BYTES // diverseBefore
                + Float.BYTES; // shortEdges
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
}
