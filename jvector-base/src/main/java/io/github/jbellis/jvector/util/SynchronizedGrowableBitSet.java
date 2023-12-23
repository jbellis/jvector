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

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A thread-safe {@link BitSet} implementation that grows as needed to accommodate set(index) calls. When it
 * does so, it will grow its internal storage multiplicatively, assuming that more growth will be
 * needed in the future. This is the important difference from FixedBitSet + FBS.ensureCapacity,
 * which grows the minimum necessary each time.
 *
 * @see GrowableBitSet
 */
public class SynchronizedGrowableBitSet extends BitSet {

  private final java.util.BitSet bitSet;
  private final Lock lock = new ReentrantLock();

  public SynchronizedGrowableBitSet(java.util.BitSet bitSet) {
    this.bitSet = bitSet;
  }

  public SynchronizedGrowableBitSet(int initialBits) {
    this.bitSet = new java.util.BitSet(initialBits);
  }

  @Override
  public void clear(int index) {
    lock.lock();
    try {
      bitSet.clear(index);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public void clear() {
    lock.lock();
    try {
      bitSet.clear();
    } finally {
      lock.unlock();
    }
  }

  @Override
  public boolean get(int index) {
    lock.lock();
    try {
      return bitSet.get(index);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public boolean getAndSet(int index) {
    lock.lock();
    try {
      boolean v = get(index);
      set(index);
      return v;
    } finally {
      lock.unlock();
    }
  }

  @Override
  public int length() {
    lock.lock();
    try {
      return bitSet.length();
    } finally {
      lock.unlock();
    }
  }

  @Override
  public void set(int i) {
    lock.lock();
    try {
      bitSet.set(i);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public void clear(int startIndex, int endIndex) {
    lock.lock();
    try {
      if (startIndex == 0 && endIndex == bitSet.length()) {
        bitSet.clear();
        return;
      } else if (startIndex >= endIndex) {
        return;
      }
      bitSet.clear(startIndex, endIndex);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public int cardinality() {
    lock.lock();
    try {
      return bitSet.cardinality();
    } finally {
      lock.unlock();
    }
  }

  @Override
  public int approximateCardinality() {
    lock.lock();
    try {
      return bitSet.cardinality();
    } finally {
      lock.unlock();
    }
  }

  @Override
  public int prevSetBit(int index) {
    lock.lock();
    try {
      return bitSet.previousSetBit(index);
    } finally {
      lock.unlock();
    }
  }

  @Override
  public int nextSetBit(int i) {
    lock.lock();
    try {
      int next = bitSet.nextSetBit(i);
      if (next == -1) {
        next = DocIdSetIterator.NO_MORE_DOCS;
      }
      return next;
    } finally {
      lock.unlock();
    }
  }

  @Override
  public long ramBytesUsed() {
    throw new UnsupportedOperationException();
  }
}
