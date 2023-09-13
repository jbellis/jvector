package com.github.jbellis.jvector.util;

/**
 * A {@link BitSet} implementation that grows as needed to accommodate set(index) calls. When it
 * does so, it will grow its internal storage multiplicatively, assuming that more growth will be
 * needed in the future. This is the important difference from FixedBitSet + FBS.ensureCapacity,
 * which grows the minimum necessary each time.
 */
public class GrowableBitSet extends BitSet {

  private final java.util.BitSet bitSet;

  public GrowableBitSet(java.util.BitSet bitSet) {
    this.bitSet = bitSet;
  }

  public GrowableBitSet(int initialBits) {
    this.bitSet = new java.util.BitSet(initialBits);
  }

  @Override
  public void clear(int index) {
    bitSet.clear(index);
  }

  @Override
  public void clear() {
    bitSet.clear();
  }

  @Override
  public boolean get(int index) {
    return bitSet.get(index);
  }

  @Override
  public boolean getAndSet(int index) {
    boolean v = get(index);
    set(index);
    return v;
  }

  @Override
  public int length() {
    return bitSet.length();
  }

  @Override
  public void set(int i) {
    bitSet.set(i);
  }

  @Override
  public void clear(int startIndex, int endIndex) {
    if (startIndex == 0 && endIndex == bitSet.length()) {
      bitSet.clear();
      return;
    } else if (startIndex >= endIndex) {
      return;
    }
    bitSet.clear(startIndex, endIndex);
  }

  @Override
  public int cardinality() {
    return bitSet.cardinality();
  }

  @Override
  public int approximateCardinality() {
    return bitSet.cardinality();
  }

  @Override
  public int prevSetBit(int index) {
    return bitSet.previousSetBit(index);
  }

  @Override
  public int nextSetBit(int i) {
    int next = bitSet.nextSetBit(i);
    if (next == -1) {
      next = DocIdSetIterator.NO_MORE_DOCS;
    }
    return next;
  }

  @Override
  public long ramBytesUsed() {
    throw new UnsupportedOperationException();
  }
}
