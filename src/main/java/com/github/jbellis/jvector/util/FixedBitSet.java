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
package com.github.jbellis.jvector.util;

import java.util.Arrays;

import com.github.jbellis.jvector.graph.NodesIterator;

/**
 * BitSet of fixed length (numBits), backed by accessible ({@link #getBits}) long[], accessed with
 * an int index, implementing {@link Bits}.
 */
public final class FixedBitSet extends BitSet {
    private final long[] bits; // Array of longs holding the bits
    private final int numBits; // The number of bits in use
    private final int numWords; // The exact number of longs needed to hold numBits (<= bits.length)

    /**
     * If the given {@link FixedBitSet} is large enough to hold {@code numBits+1}, returns the given
     * bits, otherwise returns a new {@link FixedBitSet} which can hold the requested number of bits.
     *
     * <p><b>NOTE:</b> the returned bitset reuses the underlying {@code long[]} of the given {@code
     * bits} if possible. Also, calling {@link #length()} on the returned bits may return a value
     * greater than {@code numBits}.
     */
    public static FixedBitSet ensureCapacity(FixedBitSet bits, int numBits) {
        if (numBits < bits.numBits) {
            return bits;
        } else {
            // Depends on the ghost bits being clear!
            // (Otherwise, they may become visible in the new instance)
            int numWords = bits2words(numBits);
            long[] arr = bits.getBits();
            if (numWords >= arr.length) {
                arr = ArrayUtil.grow(arr, numWords + 1);
            }
            return new FixedBitSet(arr, arr.length << 6);
        }
    }

    /** returns the number of 64 bit words it would take to hold numBits */
    public static int bits2words(int numBits) {
        // I.e.: get the word-offset of the last bit and add one (make sure to use >> so 0
        // returns 0!)
        return ((numBits - 1) >> 6) + 1;
    }

    /**
     * Returns the popcount or cardinality of the intersection of the two sets. Neither set is
     * modified.
     */
    public static long intersectionCount(FixedBitSet a, FixedBitSet b) {
        // Depends on the ghost bits being clear!
        long tot = 0;
        final int numCommonWords = Math.min(a.numWords, b.numWords);
        for (int i = 0; i < numCommonWords; ++i) {
            tot += Long.bitCount(a.bits[i] & b.bits[i]);
        }
        return tot;
    }

    /** Returns the popcount or cardinality of the union of the two sets. Neither set is modified. */
    public static long unionCount(FixedBitSet a, FixedBitSet b) {
        // Depends on the ghost bits being clear!
        long tot = 0;
        final int numCommonWords = Math.min(a.numWords, b.numWords);
        for (int i = 0; i < numCommonWords; ++i) {
            tot += Long.bitCount(a.bits[i] | b.bits[i]);
        }
        for (int i = numCommonWords; i < a.numWords; ++i) {
            tot += Long.bitCount(a.bits[i]);
        }
        for (int i = numCommonWords; i < b.numWords; ++i) {
            tot += Long.bitCount(b.bits[i]);
        }
        return tot;
    }

    /**
     * Returns the popcount or cardinality of "a and not b" or "intersection(a, not(b))". Neither set
     * is modified.
     */
    public static long andNotCount(FixedBitSet a, FixedBitSet b) {
        // Depends on the ghost bits being clear!
        long tot = 0;
        final int numCommonWords = Math.min(a.numWords, b.numWords);
        for (int i = 0; i < numCommonWords; ++i) {
            tot += Long.bitCount(a.bits[i] & ~b.bits[i]);
        }
        for (int i = numCommonWords; i < a.numWords; ++i) {
            tot += Long.bitCount(a.bits[i]);
        }
        return tot;
    }

    /**
     * Creates a new LongBitSet. The internally allocated long array will be exactly the size needed
     * to accommodate the numBits specified.
     *
     * @param numBits the number of bits needed
     */
    public FixedBitSet(int numBits) {
        this.numBits = numBits;
        bits = new long[bits2words(numBits)];
        numWords = bits.length;
    }

    /**
     * Creates a new LongBitSet using the provided long[] array as backing store. The storedBits array
     * must be large enough to accommodate the numBits specified, but may be larger. In that case the
     * 'extra' or 'ghost' bits must be clear (or they may provoke spurious side-effects)
     *
     * @param storedBits the array to use as backing store
     * @param numBits the number of bits actually needed
     */
    public FixedBitSet(long[] storedBits, int numBits) {
        this.numWords = bits2words(numBits);
        if (numWords > storedBits.length) {
            throw new IllegalArgumentException(
                    "The given long array is too small  to hold " + numBits + " bits");
        }
        this.numBits = numBits;
        this.bits = storedBits;

        assert verifyGhostBitsClear();
    }

    @Override
    public void clear() {
        Arrays.fill(bits, 0L);
    }

    /**
     * Checks if the bits past numBits are clear. Some methods rely on this implicit assumption:
     * search for "Depends on the ghost bits being clear!"
     *
     * @return true if the bits past numBits are clear.
     */
    private boolean verifyGhostBitsClear() {
        for (int i = numWords; i < bits.length; i++) {
            if (bits[i] != 0) return false;
        }

        if ((numBits & 0x3f) == 0) return true;

        long mask = -1L << numBits;

        return (bits[numWords - 1] & mask) == 0;
    }

    @Override
    public int length() {
        return numBits;
    }

    /** Expert. */
    public long[] getBits() {
        return bits;
    }

    /**
     * Returns number of set bits. NOTE: this visits every long in the backing bits array, and the
     * result is not internally cached!
     */
    @Override
    public int cardinality() {
        // Depends on the ghost bits being clear!
        long tot = 0;
        for (int i = 0; i < numWords; ++i) {
            tot += Long.bitCount(bits[i]);
        }
        return Math.toIntExact(tot);
    }

    @Override
    public int approximateCardinality() {
        // Naive sampling: compute the number of bits that are set on the first 16 longs every 1024
        // longs and scale the result by 1024/16.
        // This computes the pop count on ranges instead of single longs in order to take advantage of
        // vectorization.

        final int rangeLength = 16;
        final int interval = 1024;

        if (numWords <= interval) {
            return cardinality();
        }

        long popCount = 0;
        int maxWord;
        for (maxWord = 0; maxWord + interval < numWords; maxWord += interval) {
            for (int i = 0; i < rangeLength; ++i) {
                popCount += Long.bitCount(bits[maxWord + i]);
            }
        }

        popCount *= (interval / rangeLength) * numWords / maxWord;
        return (int) popCount;
    }

    @Override
    public boolean get(int index) {
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int i = index >> 6; // div 64
        // signed shift will keep a negative index and force an
        // array-index-out-of-bounds-exception, removing the need for an explicit check.
        long bitmask = 1L << index;
        return (bits[i] & bitmask) != 0;
    }

    @Override
    public void set(int index) {
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int wordNum = index >> 6; // div 64
        long bitmask = 1L << index;
        bits[wordNum] |= bitmask;
    }

    @Override
    public boolean getAndSet(int index) {
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int wordNum = index >> 6; // div 64
        long bitmask = 1L << index;
        boolean val = (bits[wordNum] & bitmask) != 0;
        bits[wordNum] |= bitmask;
        return val;
    }

    @Override
    public void clear(int index) {
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int wordNum = index >> 6;
        long bitmask = 1L << index;
        bits[wordNum] &= ~bitmask;
    }

    public boolean getAndClear(int index) {
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int wordNum = index >> 6; // div 64
        long bitmask = 1L << index;
        boolean val = (bits[wordNum] & bitmask) != 0;
        bits[wordNum] &= ~bitmask;
        return val;
    }

    @Override
    public int nextSetBit(int index) {
        // Depends on the ghost bits being clear!
        assert index >= 0 && index < numBits : "index=" + index + ", numBits=" + numBits;
        int i = index >> 6;
        long word = bits[i] >> index; // skip all the bits to the right of index

        if (word != 0) {
            return index + Long.numberOfTrailingZeros(word);
        }

        while (++i < numWords) {
            word = bits[i];
            if (word != 0) {
                return (i << 6) + Long.numberOfTrailingZeros(word);
            }
        }

        return NodesIterator.NO_MORE_NEIGHBORS;
    }

    @Override
    public int prevSetBit(int index) {
        assert index >= 0 && index < numBits : "index=" + index + " numBits=" + numBits;
        int i = index >> 6;
        final int subIndex = index & 0x3f; // index within the word
        long word = (bits[i] << (63 - subIndex)); // skip all the bits to the left of index

        if (word != 0) {
            return (i << 6) + subIndex - Long.numberOfLeadingZeros(word); // See LUCENE-3197
        }

        while (--i >= 0) {
            word = bits[i];
            if (word != 0) {
                return (i << 6) + 63 - Long.numberOfLeadingZeros(word);
            }
        }

        return -1;
    }

    /** this = this OR other */
    public void or(FixedBitSet other) {
        or(0, other.bits, other.numWords);
    }

    private void or(final int otherOffsetWords, FixedBitSet other) {
        or(otherOffsetWords, other.bits, other.numWords);
    }

    private void or(final int otherOffsetWords, final long[] otherArr, final int otherNumWords) {
        assert otherNumWords + otherOffsetWords <= numWords
                : "numWords=" + numWords + ", otherNumWords=" + otherNumWords;
        int pos = Math.min(numWords - otherOffsetWords, otherNumWords);
        final long[] thisArr = this.bits;
        while (--pos >= 0) {
            thisArr[pos + otherOffsetWords] |= otherArr[pos];
        }
    }

    /** this = this XOR other */
    public void xor(FixedBitSet other) {
        xor(other.bits, other.numWords);
    }

    private void xor(long[] otherBits, int otherNumWords) {
        assert otherNumWords <= numWords : "numWords=" + numWords + ", other.numWords=" + otherNumWords;
        final long[] thisBits = this.bits;
        int pos = Math.min(numWords, otherNumWords);
        while (--pos >= 0) {
            thisBits[pos] ^= otherBits[pos];
        }
    }

    /** returns true if the sets have any elements in common */
    public boolean intersects(FixedBitSet other) {
        // Depends on the ghost bits being clear!
        int pos = Math.min(numWords, other.numWords);
        while (--pos >= 0) {
            if ((bits[pos] & other.bits[pos]) != 0) return true;
        }
        return false;
    }

    /** this = this AND other */
    public void and(FixedBitSet other) {
        and(other.bits, other.numWords);
    }

    private void and(final long[] otherArr, final int otherNumWords) {
        final long[] thisArr = this.bits;
        int pos = Math.min(this.numWords, otherNumWords);
        while (--pos >= 0) {
            thisArr[pos] &= otherArr[pos];
        }
        if (this.numWords > otherNumWords) {
            Arrays.fill(thisArr, otherNumWords, this.numWords, 0L);
        }
    }

    /** this = this AND NOT other */
    public void andNot(FixedBitSet other) {
        andNot(0, other.bits, other.numWords);
    }

    private void andNot(final int otherOffsetWords, FixedBitSet other) {
        andNot(otherOffsetWords, other.bits, other.numWords);
    }

    private void andNot(final int otherOffsetWords, final long[] otherArr, final int otherNumWords) {
        int pos = Math.min(numWords - otherOffsetWords, otherNumWords);
        final long[] thisArr = this.bits;
        while (--pos >= 0) {
            thisArr[pos + otherOffsetWords] &= ~otherArr[pos];
        }
    }

    /**
     * Scans the backing store to check if all bits are clear. The method is deliberately not called
     * "isEmpty" to emphasize it is not low cost (as isEmpty usually is).
     *
     * @return true if all bits are clear.
     */
    public boolean scanIsEmpty() {
        // This 'slow' implementation is still faster than any external one could be
        // (e.g.: (bitSet.length() == 0 || bitSet.nextSetBit(0) == -1))
        // especially for small BitSets
        // Depends on the ghost bits being clear!
        final int count = numWords;

        for (int i = 0; i < count; i++) {
            if (bits[i] != 0) return false;
        }

        return true;
    }

    /**
     * Flips a range of bits
     *
     * @param startIndex lower index
     * @param endIndex one-past the last bit to flip
     */
    public void flip(int startIndex, int endIndex) {
        assert startIndex >= 0 && startIndex < numBits;
        assert endIndex >= 0 && endIndex <= numBits;
        if (endIndex <= startIndex) {
            return;
        }

        int startWord = startIndex >> 6;
        int endWord = (endIndex - 1) >> 6;

        /* Grrr, java shifting uses only the lower 6 bits of the count so -1L>>>64 == -1
         * for that reason, make sure not to use endmask if the bits to flip will
         * be zero in the last word (redefine endWord to be the last changed...)
         * long startmask = -1L << (startIndex & 0x3f);     // example: 11111...111000
         * long endmask = -1L >>> (64-(endIndex & 0x3f));   // example: 00111...111111
         */

        long startmask = -1L << startIndex;
        long endmask = -1L >>> -endIndex;

        if (startWord == endWord) {
            bits[startWord] ^= (startmask & endmask);
            return;
        }

        bits[startWord] ^= startmask;

        for (int i = startWord + 1; i < endWord; i++) {
            bits[i] = ~bits[i];
        }

        bits[endWord] ^= endmask;
    }

    /** Flip the bit at the provided index. */
    public void flip(int index) {
        assert index >= 0 && index < numBits : "index=" + index + " numBits=" + numBits;
        int wordNum = index >> 6; // div 64
        long bitmask = 1L << index; // mod 64 is implicit
        bits[wordNum] ^= bitmask;
    }

    /**
     * Sets a range of bits
     *
     * @param startIndex lower index
     * @param endIndex one-past the last bit to set
     */
    public void set(int startIndex, int endIndex) {
        assert startIndex >= 0 && startIndex < numBits
                : "startIndex=" + startIndex + ", numBits=" + numBits;
        assert endIndex >= 0 && endIndex <= numBits : "endIndex=" + endIndex + ", numBits=" + numBits;
        if (endIndex <= startIndex) {
            return;
        }

        int startWord = startIndex >> 6;
        int endWord = (endIndex - 1) >> 6;

        long startmask = -1L << startIndex;
        long endmask = -1L >>> -endIndex;

        if (startWord == endWord) {
            bits[startWord] |= (startmask & endmask);
            return;
        }

        bits[startWord] |= startmask;
        Arrays.fill(bits, startWord + 1, endWord, -1L);
        bits[endWord] |= endmask;
    }

    @Override
    public void clear(int startIndex, int endIndex) {
        assert startIndex >= 0 && startIndex < numBits
                : "startIndex=" + startIndex + ", numBits=" + numBits;
        assert endIndex >= 0 && endIndex <= numBits : "endIndex=" + endIndex + ", numBits=" + numBits;
        if (endIndex <= startIndex) {
            return;
        }

        int startWord = startIndex >> 6;
        int endWord = (endIndex - 1) >> 6;

        long startmask = -1L << startIndex;
        long endmask = -1L >>> -endIndex;

        // invert masks since we are clearing
        startmask = ~startmask;
        endmask = ~endmask;

        if (startWord == endWord) {
            bits[startWord] &= (startmask | endmask);
            return;
        }

        bits[startWord] &= startmask;
        Arrays.fill(bits, startWord + 1, endWord, 0L);
        bits[endWord] &= endmask;
    }

    @Override
    public FixedBitSet clone() {
        long[] bits = new long[this.bits.length];
        System.arraycopy(this.bits, 0, bits, 0, numWords);
        return new FixedBitSet(bits, numBits);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FixedBitSet)) {
            return false;
        }
        FixedBitSet other = (FixedBitSet) o;
        if (numBits != other.numBits) {
            return false;
        }
        // Depends on the ghost bits being clear!
        return Arrays.equals(bits, other.bits);
    }

    @Override
    public int hashCode() {
        // Depends on the ghost bits being clear!
        long h = 0;
        for (int i = numWords; --i >= 0; ) {
            h ^= bits[i];
            h = (h << 1) | (h >>> 63); // rotate left
        }
        // fold leftmost bits into right and add a constant to prevent
        // empty sets from returning 0, which is too common.
        return (int) ((h >> 32) ^ h) + 0x98761234;
    }

    /** Make a copy of the given bits. */
    public static FixedBitSet copyOf(Bits bits) {
        if (bits instanceof FixedBitSet) {
            return ((FixedBitSet) bits).clone();
        } else {
            int length = bits.length();
            FixedBitSet bitSet = new FixedBitSet(length);
            bitSet.set(0, length);
            for (int i = 0; i < length; ++i) {
                if (bits.get(i) == false) {
                    bitSet.clear(i);
                }
            }
            return bitSet;
        }
    }
}
