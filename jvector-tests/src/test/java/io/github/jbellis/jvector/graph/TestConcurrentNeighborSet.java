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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestConcurrentNeighborSet extends RandomizedTest {

  private static void validateSortedByScore(NodeArray na) {
    for (int i = 0; i < na.size() - 1; i++) {
      assertTrue(na.score[i] >= na.score[i + 1]);
    }
  }

  @Test
  public void testInsertDiverse() {
    // set up scoreBetween
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new GraphIndexTestCase.CircularFloatVectorValues(10);
    var vectorsCopy = vectors.copy();
    var candidates = new NodeArray(10);
    NodeSimilarity scoreBetween = a -> {
      return (NodeSimilarity.ExactScoreFunction) b -> similarityFunction.compare(vectors.vectorValue(a), vectorsCopy.vectorValue(b));
    };
    // fill candidates with all the nodes except 7
    IntStream.range(0, 10)
        .filter(i -> i != 7)
        .forEach(
            i -> {
              candidates.insertSorted(i, scoreBetween.score(7, i));
            });
    assert candidates.size() == 9;

    // only nodes 6 and 8 are diverse wrt 7
    var neighbors = new ConcurrentNeighborSet(7, 3, scoreBetween);
    var empty = new NodeArray(0);
    neighbors.insertDiverse(candidates, empty);
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
    validateSortedByScore(neighbors.getCurrent());
  }

  @Test
  public void testInsertDiverseConcurrent() {
    // set up scoreBetween
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new GraphIndexTestCase.CircularFloatVectorValues(10);
    var vectorsCopy = vectors.copy();
    var natural = new NodeArray(10);
    var concurrent = new NodeArray(10);
    NodeSimilarity scoreBetween = a -> {
      return (NodeSimilarity.ExactScoreFunction) b -> similarityFunction.compare(vectors.vectorValue(a), vectorsCopy.vectorValue(b));
    };
    // "natural" candidates are [0..7), "concurrent" are [8..10)
    IntStream.range(0, 7)
        .forEach(
            i -> {
              natural.insertSorted(i, scoreBetween.score(7, i));
            });
    IntStream.range(8, 10)
        .forEach(
            i -> {
              concurrent.insertSorted(i, scoreBetween.score(7, i));
            });

    // only nodes 6 and 8 are diverse wrt 7
    var neighbors = new ConcurrentNeighborSet(7, 3, scoreBetween);
    neighbors.insertDiverse(natural, concurrent);
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
    validateSortedByScore(neighbors.getCurrent());
  }

  @Test
  public void testInsertDiverseRetainsNatural() {
    // set up scoreBetween
    var vectors = new GraphIndexTestCase.CircularFloatVectorValues(10);
    var vectorsCopy = vectors.copy();
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    NodeSimilarity scoreBetween = a -> {
      return (NodeSimilarity.ExactScoreFunction) b -> similarityFunction.compare(vectors.vectorValue(a), vectorsCopy.vectorValue(b));
    };

    // check that the new neighbor doesn't replace the existing one (since both are diverse, and the max degree accommodates both)
    var cna = new NodeArray(1);
    cna.addInOrder(6, scoreBetween.score(7, 6));

    var cna2 = new NodeArray(1);
    cna2.addInOrder(8, scoreBetween.score(7, 6));

    var neighbors = new ConcurrentNeighborSet(7, 3, scoreBetween, 1.0f, cna);
    neighbors.insertDiverse(cna2, NodeArray.EMPTY);
    assertEquals(2, neighbors.size());
  }

  @Test
  public void testNoDuplicatesDescOrder() {
    NodeArray cna = new NodeArray(5);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 9.0f);
    cna.insertSorted(3, 8.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 8.0f); // This is also a duplicate
    Assert.assertArrayEquals(new int[] {1, 2, 3}, ArrayUtil.copyOfSubArray(cna.node(), 0, cna.size()));
    assertArrayEquals(
        new float[] {10.0f, 9.0f, 8.0f}, ArrayUtil.copyOfSubArray(cna.score, 0, cna.size()), 0.01f);
    validateSortedByScore(cna);
  }

  @Test
  public void testNoDuplicatesSameScores() {
    NodeArray cna = new NodeArray(5);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 10.0f);
    cna.insertSorted(3, 10.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 10.0f); // This is also a duplicate
    assertArrayEquals(new int[] {1, 2, 3}, ArrayUtil.copyOfSubArray(cna.node(), 0, cna.size()));
    assertArrayEquals(new float[] {10.0f, 10.0f, 10.0f}, ArrayUtil.copyOfSubArray(cna.score, 0, cna.size()), 0.01f);
    validateSortedByScore(cna);
  }

  @Test
  public void testMergeCandidatesSimple() {
    var arr1 = new NodeArray(1);
    arr1.addInOrder(1, 1.0f);

    var arr2 = new NodeArray(1);
    arr2.addInOrder(0, 2.0f);

    var merged = ConcurrentNeighborSet.mergeNeighbors(arr1, arr2);
    // Expected result: [0, 1]
    assertEquals(2, merged.size());
    assertArrayEquals(new int[] {0, 1}, Arrays.copyOf(merged.node(), 2));

    arr1 = new NodeArray(3);
    arr1.addInOrder(3, 3.0f);
    arr1.addInOrder(2, 2.0f);
    arr1.addInOrder(1, 1.0f);

    arr2 = new NodeArray(3);
    arr2.addInOrder(4, 4.0f);
    arr2.addInOrder(2, 2.0f);
    arr2.addInOrder(1, 1.0f);

    merged = ConcurrentNeighborSet.mergeNeighbors(arr1, arr2);
    // Expected result: [4, 3, 2, 1]
    assertEquals(4, merged.size());
    assertArrayEquals(new int[] {4, 3, 2, 1}, Arrays.copyOf(merged.node(), 4));
    assertArrayEquals(new float[] {4.0f, 3.0f, 2.0f, 1.0f}, Arrays.copyOf(merged.score(), 4), 0.0f);

    // Testing boundary conditions
    arr1 = new NodeArray(2);
    arr1.addInOrder(3, 3.0f);
    arr1.addInOrder(2, 2.0f);

    arr2 = new NodeArray(1);
    arr2.addInOrder(2, 2.0f);

    merged = ConcurrentNeighborSet.mergeNeighbors(arr1, arr2);
    // Expected result: [3, 2]
    assertEquals(2, merged.size());
    assertArrayEquals(new int[] {3, 2}, Arrays.copyOf(merged.node(), 2));
    assertArrayEquals(new float[] {3.0f, 2.0f}, Arrays.copyOf(merged.score(), 2), 0.0f);
    validateSortedByScore(merged);
  }

  private void testMergeCandidatesOnce() {
    // test merge emphasizing dealing with tied scores
    int maxSize = 1 + getRandom().nextInt(5);

    // fill arr1 with nodes from 0..size, with random scores assigned (so random order of nodes)
    NodeArray arr1 = new NodeArray(maxSize);
    int a1Size = getRandom().nextBoolean() ? maxSize : 1 + getRandom().nextInt(maxSize);
    for (int i = 0; i < a1Size; i++) {
      arr1.insertSorted(i, getRandom().nextFloat());
    }

    // arr2 contains either
    // -- an exact duplicates of the corresponding node in arr1, or
    // -- a random score chosen from arr1
    // this is designed to maximize the need for correct handling of corner cases in the merge
    NodeArray arr2 = new NodeArray(maxSize);
    int a2Size = getRandom().nextBoolean() ? maxSize : 1 + getRandom().nextInt(maxSize);
    for (int i = 0; i < a2Size; i++) {
      if (i < a1Size && getRandom().nextBoolean()) {
        // duplicate entry
        int j = getRandom().nextInt(a1Size);
        if (!arr2.contains(arr1.node[j])) {
          arr2.insertSorted(arr1.node[j], arr1.score[j]);
        }
      } else {
        // duplicate just score
        float score;
        if (getRandom().nextBoolean()) {
          score = getRandom().nextFloat();
        } else {
          score = arr1.score[getRandom().nextInt(a1Size)];
        }
        arr2.insertSorted(i + arr1.size, score);
      }
    }

    // merge!
    var merged = ConcurrentNeighborSet.mergeNeighbors(arr1, arr2);

    // sanity check
    assert merged.size <= arr1.size() + arr2.size();
    assert merged.size >= Math.max(arr1.size(), arr2.size());
    var uniqueNodes = new HashSet<>();

    // results should be sorted by score, and not contain duplicates
    for (int i = 0; i < merged.size - 1; i++) {
      assertTrue(merged.score[i] >= merged.score[i + 1]);
      assertTrue(uniqueNodes.add(merged.node[i]));
    }
    assertTrue(uniqueNodes.add(merged.node[merged.size - 1]));

    // results should contain all the nodes that were in the source arrays
    for (int i = 0; i < arr1.size(); i++) {
      assertTrue(String.format("%s missing%na1: %s%na2: %s%nmerged: %s%n",
                               arr1.node[i],
                               Arrays.toString(arr1.node),
                               Arrays.toString(arr2.node),
                               Arrays.toString(merged.node)),
                 uniqueNodes.contains(arr1.node[i]));
    }
    for (int i = 0; i < arr2.size(); i++) {
        assertTrue(String.format("%s missing%na1: %s%na2: %s%nmerged: %s%n",
                                 arr2.node[i],
                                 Arrays.toString(arr1.node),
                                 Arrays.toString(arr2.node),
                                 Arrays.toString(merged.node)),
                     uniqueNodes.contains(arr2.node[i]));
    }
  }

  @Test
  public void testMergeCandidatesRandom() {
    for (int i = 0; i < 10000; i++) {
      testMergeCandidatesOnce();
    }
  }
}
