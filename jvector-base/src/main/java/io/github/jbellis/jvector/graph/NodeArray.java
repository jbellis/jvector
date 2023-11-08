/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.Bits;

import java.util.Arrays;

/**
 * NodeArray encodes nodeids and their scores relative to some other element 
 * (a query vector, or another graph node) as a pair of growable arrays. 
 * Nodes are arranged in the sorted order of their scores in descending order,
 * i.e. the most-similar nodes are first.
 */
public class NodeArray {
    public static final NodeArray EMPTY = new NodeArray(0);

    protected int size;
    float[] score;
    int[] node;

    public NodeArray(int maxSize) {
        node = new int[maxSize];
        score = new float[maxSize];
    }

    /**
     * Add a new node to the NodeArray. The new node must be worse than all previously stored
     * nodes.
     */
    public void addInOrder(int newNode, float newScore) {
        if (size == node.length) {
            node = ArrayUtil.grow(node);
            score = ArrayUtil.growExact(score, node.length);
        }
        if (size > 0) {
            float previousScore = score[size - 1];
            assert ((previousScore >= newScore))
                    : "Nodes are added in the incorrect order! Comparing "
                    + newScore
                    + " to "
                    + Arrays.toString(ArrayUtil.copyOfSubArray(score, 0, size));
        }
        node[size] = newNode;
        score[size] = newScore;
        ++size;
    }

    /**
     * Add a new node to the NodeArray into a correct sort position according to its score.
     * Duplicate node + score pairs are ignored.
     *
     * @return true if the new node + score pair did not already exist
     */
    public boolean insertSorted(int newNode, float newScore) {
        if (size == node.length) {
            growArrays();
        }
        int insertionPoint = descSortFindRightMostInsertionPoint(newScore);
        if (duplicateExistsNear(insertionPoint, newNode, newScore)) {
            return false;
        }

        System.arraycopy(node, insertionPoint, node, insertionPoint + 1, size - insertionPoint);
        System.arraycopy(score, insertionPoint, score, insertionPoint + 1, size - insertionPoint);
        node[insertionPoint] = newNode;
        score[insertionPoint] = newScore;
        ++size;
        return true;
    }

    private boolean duplicateExistsNear(int insertionPoint, int newNode, float newScore) {
        // Check to the left
        for (int i = insertionPoint - 1; i >= 0 && score[i] == newScore; i--) {
            if (node[i] == newNode) {
                return true;
            }
        }

        // Check to the right
        for (int i = insertionPoint; i < size && score[i] == newScore; i++) {
            if (node[i] == newNode) {
                return true;
            }
        }

        return false;
    }

    /**
     * Retains only the elements in the current NodeArray whose corresponding index
     * is set in the given BitSet.
     * <p>
     * This modifies the array in place, preserving the relative order of the elements retained.
     * <p>
     *
     * @param selected A BitSet where the bit at index i is set if the i-th element should be retained.
     *                 (Thus, the elements of selected represent positions in the NodeArray, NOT node ids.)
     */
    public void retain(Bits selected) {
        int writeIdx = 0; // index for where to write the next retained element

        for (int readIdx = 0; readIdx < size; readIdx++) {
            if (selected.get(readIdx)) {
                if (writeIdx != readIdx) {
                    // Move the selected entries to the front while maintaining their relative order
                    node[writeIdx] = node[readIdx];
                    score[writeIdx] = score[readIdx];
                }
                // else { we haven't created any gaps in the backing arrays yet, so we don't need to move anything }
                writeIdx++;
            }
        }

        size = writeIdx;
    }

    public NodeArray copy() {
        NodeArray copy = new NodeArray(node.length);
        copy.size = size;
        System.arraycopy(node, 0, copy.node, 0, size);
        System.arraycopy(score, 0, copy.score, 0, size);
        return copy;
    }

    protected final void growArrays() {
        node = ArrayUtil.grow(node);
        score = ArrayUtil.growExact(score, node.length);
    }

    public int size() {
        return size;
    }

    /**
     * Direct access to the internal list of node ids; provided for efficient writing of the graph
     */
    public int[] node() {
        return node;
    }

    public float[] score() {
        return score;
    }

    public void clear() {
        size = 0;
    }

    public void removeLast() {
        size--;
    }

    public void removeIndex(int idx) {
        System.arraycopy(node, idx + 1, node, idx, size - idx - 1);
        System.arraycopy(score, idx + 1, score, idx, size - idx - 1);
        size--;
    }

    @Override
    public String toString() {
        return "NodeArray[" + size + "]";
    }

    protected final int descSortFindRightMostInsertionPoint(float newScore) {
        int start = 0;
        int end = size - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (score[mid] < newScore) end = mid - 1;
            else start = mid + 1;
        }
        return start;
    }

    /**
     * Caution! This performs a linear scan.
     */
    @VisibleForTesting
    boolean contains(int node) {
        for (int i = 0; i < size; i++) {
            if (this.node[i] == node) {
                return true;
            }
        }
        return false;
    }

    @VisibleForTesting
    int[] copyDenseNodes() {
        return Arrays.copyOf(node, size);
    }
}
