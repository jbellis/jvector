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

package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.TestUtil;
import org.junit.Assert;
import org.junit.Test;

public class TestAtomicFixedBitSet extends BaseBitSetTestCase<AtomicFixedBitSet> {

  @Override
  public AtomicFixedBitSet copyOf(BitSet bs, int length) {
    final AtomicFixedBitSet set = new AtomicFixedBitSet(length);
    for (int doc = bs.nextSetBit(0); doc != DocIdSetIterator.NO_MORE_DOCS; doc = bs.nextSetBit(doc + 1)) {
      set.set(doc);
    }
    return set;
  }

  @SuppressWarnings("NarrowCalculation")
  @Test
  public void testApproximateCardinality() {
    // The approximate cardinality works in such a way that it should be pretty accurate on a bitset
    // whose bits are uniformly distributed.
    final AtomicFixedBitSet set = new AtomicFixedBitSet(TestUtil.nextInt(random(), 100_000, 200_000));
    final int first = random().nextInt(10);
    final int interval = TestUtil.nextInt(random(), 10, 20);
    for (int i = first; i < set.length(); i += interval) {
      set.set(i);
    }
    final int cardinality = set.cardinality();
    Assert.assertEquals(cardinality, set.approximateCardinality(), cardinality / 20); // 5% error at most
  }
}
