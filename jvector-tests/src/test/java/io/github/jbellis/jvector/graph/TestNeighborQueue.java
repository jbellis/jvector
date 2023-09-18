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

import com.carrotsearch.randomizedtesting.RandomizedTest;
import org.junit.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestNeighborQueue extends RandomizedTest {
  @Test
  public void testNeighborsProduct() {
    // make sure we have the sign correct
    NeighborQueue nn = new NeighborQueue(2, false);
    assertTrue(nn.insertWithReplacement(2, 0.5f));
    assertTrue(nn.insertWithReplacement(1, 0.2f));
    assertTrue(nn.insertWithReplacement(3, 1f));
    assertEquals(0.5f, nn.topScore(), 0);
    nn.pop();
    assertEquals(1f, nn.topScore(), 0);
    nn.pop();
  }

  @Test
  public void testNeighborsMaxHeap() {
    NeighborQueue nn = new NeighborQueue(2, true);
    assertTrue(nn.insertWithReplacement(2, 2));
    assertTrue(nn.insertWithReplacement(1, 1));
    assertFalse(nn.insertWithReplacement(3, 3));
    assertEquals(2f, nn.topScore(), 0);
    nn.pop();
    assertEquals(1f, nn.topScore(), 0);
  }

  @Test
  public void testTopMaxHeap() {
    NeighborQueue nn = new NeighborQueue(2, true);
    nn.add(1, 2);
    nn.add(2, 1);
    // lower scores are better; highest score on top
    assertEquals(2, nn.topScore(), 0);
    assertEquals(1, nn.topNode());
  }

  @Test
  public void testTopMinHeap() {
    NeighborQueue nn = new NeighborQueue(2, false);
    nn.add(1, 0.5f);
    nn.add(2, -0.5f);
    // higher scores are better; lowest score on top
    assertEquals(-0.5f, nn.topScore(), 0);
    assertEquals(2, nn.topNode());
  }

  @Test
  public void testClear() {
    NeighborQueue nn = new NeighborQueue(2, false);
    nn.add(1, 1.1f);
    nn.add(2, -2.2f);
    nn.markIncomplete();
    nn.clear();

    assertEquals(0, nn.size());
    assertFalse(nn.incomplete());
  }

  @Test
  public void testMaxSizeQueue() {
    NeighborQueue nn = new NeighborQueue(2, false);
    nn.add(1, 1);
    nn.add(2, 2);
    assertEquals(2, nn.size());
    assertEquals(1, nn.topNode());

    // insertWithOverflow does not extend the queue
    nn.insertWithReplacement(3, 3);
    assertEquals(2, nn.size());
    assertEquals(2, nn.topNode());

    // add does extend the queue beyond maxSize
    nn.add(4, 1);
    assertEquals(3, nn.size());
  }

  @Test
  public void testUnboundedQueue() {
    NeighborQueue nn = new NeighborQueue(1, true);
    float maxScore = -2;
    int maxNode = -1;
    for (int i = 0; i < 256; i++) {
      // initial size is 32
      float score = getRandom().nextFloat();
      if (score > maxScore) {
        maxScore = score;
        maxNode = i;
      }
      nn.add(i, score);
    }
    assertEquals(maxScore, nn.topScore(), 0);
    assertEquals(maxNode, nn.topNode());
  }

  @Test
  public void testInvalidArguments() {
    assertThrows(IllegalArgumentException.class, () -> new NeighborQueue(0, false));
  }

  @Test
  public void testToString() {
    assertEquals("Neighbors[0]", new NeighborQueue(2, false).toString());
  }
}
