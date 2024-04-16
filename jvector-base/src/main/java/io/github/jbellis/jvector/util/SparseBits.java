package io.github.jbellis.jvector.util;

import org.agrona.collections.IntHashSet;

/**
 * Implements the membership parts of an updatable BitSet (but not prev/next bits)
 */
public class SparseBits implements Bits {
    private final IntHashSet set = new IntHashSet();

    @Override
    public boolean get(int index) {
        return set.contains(index);
    }

    public void set(int index) {
        set.add(index);
    }

    public void clear() {
        set.clear();
    }

    public int cardinality() {
        return set.size();
    }

    @Override
    public int length() {
        throw new UnsupportedOperationException();
    }
}
