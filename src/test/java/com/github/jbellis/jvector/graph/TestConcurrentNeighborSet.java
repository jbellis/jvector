/*
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

package com.github.jbellis.jvector.graph;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.github.jbellis.jvector.graph.ConcurrentNeighborSet.ConcurrentNeighborArray;
import com.github.jbellis.jvector.util.ArrayUtil;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import org.junit.jupiter.api.Test;

import static com.github.jbellis.jvector.graph.ConcurrentNeighborSet.mergeNeighbors;
import static org.junit.Assert.*;

public class TestConcurrentNeighborSet extends RandomizedTest {
  private static final NeighborSimilarity simpleScore =
      new NeighborSimilarity() {
        @Override
        public float score(int a, int b) {
          return VectorSimilarityFunction.EUCLIDEAN.compare(new float[] {a}, new float[] {b});
        }

        @Override
        public ScoreFunction scoreProvider(int a) {
          return b -> score(a, b);
        }
      };

  private static float baseScore(int neighbor) {
    return simpleScore.score(0, neighbor);
  }

  @Test
  public void testInsertAndSize() throws IOException {
    ConcurrentNeighborSet neighbors = new ConcurrentNeighborSet(0, 2, simpleScore);
    neighbors.insert(1, baseScore(1));
    neighbors.insert(2, baseScore(2));
    assertEquals(2, neighbors.size());

    neighbors.insert(3, baseScore(3));
    assertEquals(2, neighbors.size());
  }

  @Test
  public void testRemoveLeastDiverseFromEnd() throws IOException {
    ConcurrentNeighborSet neighbors = new ConcurrentNeighborSet(0, 3, simpleScore);
    neighbors.insert(1, baseScore(1));
    neighbors.insert(2, baseScore(2));
    neighbors.insert(3, baseScore(3));
    assertEquals(3, neighbors.size());

    neighbors.insert(4, baseScore(4));
    assertEquals(3, neighbors.size());

    List<Integer> expectedValues = Arrays.asList(1, 2, 3);
    Iterator<Integer> iterator = neighbors.nodeIterator();
    for (Integer expectedValue : expectedValues) {
      assertTrue(iterator.hasNext());
      assertEquals(expectedValue, iterator.next());
    }
    assertFalse(iterator.hasNext());
  }

  @Test
  public void testInsertDiverse() {
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new GraphIndexTestCase.CircularFloatVectorValues(10);
    var vectorsCopy = vectors.copy();
    var candidates = new NeighborArray(10, true);
    NeighborSimilarity scoreBetween =
        new NeighborSimilarity() {
          @Override
          public float score(int a, int b) {
            return similarityFunction.compare(vectors.vectorValue(a), vectorsCopy.vectorValue(b));
          }

          @Override
          public ScoreFunction scoreProvider(int a) {
            return b -> score(a, b);
          }
        };
    IntStream.range(0, 10)
        .filter(i -> i != 7)
        .forEach(
            i -> {
              candidates.insertSorted(i, scoreBetween.score(7, i));
            });
    assert candidates.size() == 9;

    var neighbors = new ConcurrentNeighborSet(7, 3, scoreBetween);
    var empty = new NeighborArray(0, true);
    neighbors.insertDiverse(candidates, empty);
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
  }

  @Test
  public void testInsertDiverseConcurrent() {
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var vectors = new GraphIndexTestCase.CircularFloatVectorValues(10);
    var vectorsCopy = vectors.copy();
    var natural = new NeighborArray(10, true);
    var concurrent = new NeighborArray(10, true);
    NeighborSimilarity scoreBetween =
        new NeighborSimilarity() {
          @Override
          public float score(int a, int b) {
            return similarityFunction.compare(vectors.vectorValue(a), vectorsCopy.vectorValue(b));
          }

          @Override
          public ScoreFunction scoreProvider(int a) {
            return b -> score(a, b);
          }
        };
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

    var neighbors = new ConcurrentNeighborSet(7, 3, scoreBetween);
    neighbors.insertDiverse(natural, concurrent);
    assertEquals(2, neighbors.size());
    assert neighbors.contains(8);
    assert neighbors.contains(6);
  }

  @Test
  public void testNoDuplicatesDescOrder() {
    ConcurrentNeighborArray cna = new ConcurrentNeighborArray(5, true);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 9.0f);
    cna.insertSorted(3, 8.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 8.0f); // This is also a duplicate
    assertArrayEquals(new int[] {1, 2, 3}, ArrayUtil.copyOfSubArray(cna.node(), 0, cna.size()));
    assertArrayEquals(
        new float[] {10.0f, 9.0f, 8.0f}, ArrayUtil.copyOfSubArray(cna.score, 0, cna.size()), 0.01f);
  }

  @Test
  public void testNoDuplicatesAscOrder() {
    ConcurrentNeighborArray cna = new ConcurrentNeighborArray(5, false);
    cna.insertSorted(1, 8.0f);
    cna.insertSorted(2, 9.0f);
    cna.insertSorted(3, 10.0f);
    cna.insertSorted(1, 8.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 10.0f); // This is also a duplicate
    assertArrayEquals(new int[] {1, 2, 3}, ArrayUtil.copyOfSubArray(cna.node(), 0, cna.size()));
    assertArrayEquals(new float[] {8.0f, 9.0f, 10.0f}, ArrayUtil.copyOfSubArray(cna.score, 0, cna.size()), 0.01f);
  }

  @Test
  public void testNoDuplicatesSameScores() {
    ConcurrentNeighborArray cna = new ConcurrentNeighborArray(5, true);
    cna.insertSorted(1, 10.0f);
    cna.insertSorted(2, 10.0f);
    cna.insertSorted(3, 10.0f);
    cna.insertSorted(1, 10.0f); // This is a duplicate and should be ignored
    cna.insertSorted(3, 10.0f); // This is also a duplicate
    assertArrayEquals(new int[] {1, 2, 3}, ArrayUtil.copyOfSubArray(cna.node(), 0, cna.size()));
    assertArrayEquals(new float[] {10.0f, 10.0f, 10.0f}, ArrayUtil.copyOfSubArray(cna.score, 0, cna.size()), 0.01f);
  }

  @Test
  public void testMergeCandidatesSimple() {
    NeighborArray arr1 = new NeighborArray(3, true);
    arr1.addInOrder(3, 3.0f);
    arr1.addInOrder(2, 2.0f);
    arr1.addInOrder(1, 1.0f);

    NeighborArray arr2 = new NeighborArray(3, true);
    arr2.addInOrder(4, 4.0f);
    arr2.addInOrder(2, 2.0f);
    arr2.addInOrder(1, 1.0f);

    NeighborArray merged = mergeNeighbors(arr1, arr2);

    // Expected result: [4, 3, 2, 1]
    assertEquals(4, merged.size());
    assertArrayEquals(new int[] {4, 3, 2, 1}, Arrays.copyOf(merged.node(), 4));
    assertArrayEquals(new float[] {4.0f, 3.0f, 2.0f, 1.0f}, Arrays.copyOf(merged.score(), 4), 0.0f);

    // Testing boundary conditions
    arr1 = new NeighborArray(2, true);
    arr1.addInOrder(3, 3.0f);
    arr1.addInOrder(2, 2.0f);

    arr2 = new NeighborArray(1, true);
    arr2.addInOrder(2, 2.0f);

    merged = mergeNeighbors(arr1, arr2);

    // Expected result: [3, 2]
    assertEquals(2, merged.size());
    assertArrayEquals(new int[] {3, 2}, Arrays.copyOf(merged.node(), 2));
    assertArrayEquals(new float[] {3.0f, 2.0f}, Arrays.copyOf(merged.score(), 2), 0.0f);
  }

  private void testMergeCandidatesOnce() {
    int maxSize = 1 + getRandom().nextInt(5);

    NeighborArray arr1 = new NeighborArray(maxSize, true);
    int a1Size;
    if (getRandom().nextBoolean()) {
      a1Size = maxSize;
    } else {
      a1Size = 1 + getRandom().nextInt(maxSize);
    }
    for (int i = 0; i < a1Size; i++) {
      arr1.insertSorted(i, getRandom().nextFloat());
    }

    NeighborArray arr2 = new NeighborArray(maxSize, true);
    int a2Size;
    if (getRandom().nextBoolean()) {
      a2Size = maxSize;
    } else {
      a2Size = 1 + getRandom().nextInt(maxSize);
    }
    for (int i = 0; i < a2Size; i++) {
      if (getRandom().nextBoolean()) {
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

    var merged = mergeNeighbors(arr1, arr2);
    assert merged.size <= arr1.size() + arr2.size();
    assert merged.size >= Math.max(arr1.size(), arr2.size());
    for (int i = 0; i < merged.size - 1; i++) {
      assert merged.score[i] >= merged.score[i + 1];
    }
  }

  public void testMergeCandidatesRandom() {
    for (int i = 0; i < 10000; i++) {
      testMergeCandidatesOnce();
    }
  }
}
