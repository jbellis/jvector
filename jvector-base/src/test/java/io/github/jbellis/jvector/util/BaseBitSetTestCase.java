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

package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.LuceneTestCase;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

/** Base test case for BitSets. */
@Ignore
public abstract class BaseBitSetTestCase<T extends BitSet> extends LuceneTestCase {

  /** Create a copy of the given {@link BitSet} which has <code>length</code> bits. */
  public abstract T copyOf(BitSet bs, int length) throws IOException;

  /** Create a random set which has <code>numBitsSet</code> of its <code>numBits</code> bits set. */
  static java.util.BitSet randomSet(int numBits, int numBitsSet) {
    assert numBitsSet <= numBits;
    final java.util.BitSet set = new java.util.BitSet(numBits);
    if (numBitsSet == numBits) {
      set.set(0, numBits);
    } else {
      Random random = random();
      for (int i = 0; i < numBitsSet; ++i) {
        while (true) {
          final int o = random.nextInt(numBits);
          if (!set.get(o)) {
            set.set(o);
            break;
          }
        }
      }
    }
    return set;
  }

  /** Same as {@link #randomSet(int, int)} but given a load factor. */
  static java.util.BitSet randomSet(int numBits, float percentSet) {
    return randomSet(numBits, (int) (percentSet * numBits));
  }

  protected void assertEquals(BitSet set1, T set2, int maxDoc) {
    for (int i = 0; i < maxDoc; ++i) {
      boolean a = set1.get(i);
      boolean b = set2.get(i);
      Assert.assertEquals("Different at " + i, a, b);
    }
  }

  /** Test the {@link BitSet#cardinality()} method. */
  @Test
  public void testCardinality() throws IOException {
    final int numBits = 1 + random().nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      Assert.assertEquals(set1.cardinality(), set2.cardinality());
    }
  }

  /** Test {@link BitSet#prevSetBit(int)}. */
  @Test
  public void testPrevSetBit() throws IOException {
    final int numBits = 1 + random().nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      for (int i = 0; i < numBits; ++i) {
        Assert.assertEquals(Integer.toString(i), set1.prevSetBit(i), set2.prevSetBit(i));
      }
    }
  }

  /** Test {@link BitSet#nextSetBit(int)}. */
  @Test
  public void testNextSetBit() throws IOException {
    final int numBits = 1 + random().nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      for (int i = 0; i < numBits; ++i) {
        int i1 = set1.nextSetBit(i);
        int j = set2.nextSetBit(i);
        Assert.assertEquals(i1, j);
      }
    }
  }

  /** Test the {@link BitSet#set} method. */
  @Test
  public void testSet() throws IOException {
    Random random = random();
    final int numBits = 1 + random.nextInt(100000);
    BitSet set1 = new GrowableBitSet(randomSet(numBits, 0));
    T set2 = copyOf(set1, numBits);
    final int iters = 10000 + random.nextInt(10000);
    for (int i = 0; i < iters; ++i) {
      final int index = random.nextInt(numBits);
      set1.set(index);
      set2.set(index);
    }
    assertEquals(set1, set2, numBits);
  }

  /** Test the {@link BitSet#getAndSet} method. */
  @Test
  public void testGetAndSet() throws IOException {
    Random random = random();
    final int numBits = 1 + random.nextInt(100000);
    BitSet set1 = new GrowableBitSet(randomSet(numBits, 0));
    T set2 = copyOf(set1, numBits);
    final int iters = 10000 + random.nextInt(10000);
    for (int i = 0; i < iters; ++i) {
      final int index = random.nextInt(numBits);
      Assert.assertEquals(set1.getAndSet(index), set2.getAndSet(index));
    }
    assertEquals(set1, set2, numBits);
  }

  /** Test the {@link BitSet#clear(int)} method. */
  @Test
  public void testClear() throws IOException {
    Random random = random();
    final int numBits = 1 + random.nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      final int iters = 1 + random.nextInt(numBits * 2);
      for (int i = 0; i < iters; ++i) {
        final int index = random.nextInt(numBits);
        set1.clear(index);
        set2.clear(index);
      }
      assertEquals(set1, set2, numBits);
    }
  }

  /** Test the {@link BitSet#clear(int,int)} method. */
  @Test
  public void testClearRange() throws IOException {
    Random random = random();
    final int numBits = 1 + random.nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      final int iters = atLeast(random, 10);
      for (int i = 0; i < iters; ++i) {
        final int from = random.nextInt(numBits);
        final int to = random.nextInt(numBits + 1);
        set1.clear(from, to);
        set2.clear(from, to);
        assertEquals(set1, set2, numBits);
      }
    }
  }

  /** Test the {@link BitSet#clear()} method. */
  @Test
  public void testClearAll() throws IOException {
    Random random = random();
    final int numBits = 1 + random.nextInt(100000);
    for (float percentSet : new float[] {0, 0.01f, 0.1f, 0.5f, 0.9f, 0.99f, 1f}) {
      BitSet set1 = new GrowableBitSet(randomSet(numBits, percentSet));
      T set2 = copyOf(set1, numBits);
      final int iters = atLeast(random, 10);
      for (int i = 0; i < iters; ++i) {
        set1.clear();
        set2.clear();
        assertEquals(set1, set2, numBits);
      }
    }
  }
}
