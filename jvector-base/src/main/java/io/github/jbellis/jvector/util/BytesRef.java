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

import java.util.Arrays;

/**
 * Represents byte[], as a slice (offset + length) into an existing byte[]. The {@link #bytes}
 * member should never be null; use {@link #EMPTY_BYTES} if necessary.
 *
 * <p>{@code BytesRef} implements {@link Comparable}. The underlying byte arrays are sorted
 * lexicographically, numerically treating elements as unsigned. This is identical to Unicode
 * codepoint order.
 */
public final class BytesRef implements Comparable<BytesRef>, Cloneable {
  /** An empty byte array for convenience */
  public static final byte[] EMPTY_BYTES = new byte[0];

  /** The contents of the BytesRef. Should never be {@code null}. */
  public byte[] bytes;

  /** Offset of first valid byte. */
  public int offset;

  /** Length of used bytes. */
  public int length;

    /** This instance will directly reference bytes w/o making a copy. bytes should not be null. */
  public BytesRef(byte[] bytes, int offset, int length) {
    this.bytes = bytes;
    this.offset = offset;
    this.length = length;
    assert isValid();
  }

  /** This instance will directly reference bytes w/o making a copy. bytes should not be null */
  public BytesRef(byte[] bytes) {
    this(bytes, 0, bytes.length);
  }

  /**
   * Create a BytesRef pointing to a new array of size <code>capacity</code>. Offset and length will
   * both be zero.
   */
  public BytesRef(int capacity) {
    this.bytes = new byte[capacity];
  }

    /**
   * Expert: compares the bytes against another BytesRef, returning true if the bytes are equal.
   *
   * @param other Another BytesRef, should not be null.
   */
  public boolean bytesEquals(BytesRef other) {
    return Arrays.equals(
        this.bytes,
        this.offset,
        this.offset + this.length,
        other.bytes,
        other.offset,
        other.offset + other.length);
  }

  /**
   * Returns a shallow clone of this instance (the underlying bytes are <b>not</b> copied and will
   * be shared by both the returned object and this object.
   */
  @Override
  public BytesRef clone() {
    return new BytesRef(bytes, offset, length);
  }

  @Override
  public boolean equals(Object other) {
    if (other == null) {
      return false;
    }
    if (other instanceof BytesRef) {
      return this.bytesEquals((BytesRef) other);
    }
    return false;
  }

    /** Returns hex encoded bytes, eg [0x6c 0x75 0x63 0x65 0x6e 0x65] */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append('[');
    final int end = offset + length;
    for (int i = offset; i < end; i++) {
      if (i > offset) {
        sb.append(' ');
      }
      sb.append(Integer.toHexString(bytes[i] & 0xff));
    }
    sb.append(']');
    return sb.toString();
  }

  /** Unsigned byte order comparison */
  @Override
  public int compareTo(BytesRef other) {
    return Arrays.compareUnsigned(
        this.bytes,
        this.offset,
        this.offset + this.length,
        other.bytes,
        other.offset,
        other.offset + other.length);
  }

    /** Performs internal consistency checks. Always returns true (or throws IllegalStateException) */
  public boolean isValid() {
    if (bytes == null) {
      throw new IllegalStateException("bytes is null");
    }
    if (length < 0) {
      throw new IllegalStateException("length is negative: " + length);
    }
    if (length > bytes.length) {
      throw new IllegalStateException(
          "length is out of bounds: " + length + ",bytes.length=" + bytes.length);
    }
    if (offset < 0) {
      throw new IllegalStateException("offset is negative: " + offset);
    }
    if (offset > bytes.length) {
      throw new IllegalStateException(
          "offset out of bounds: " + offset + ",bytes.length=" + bytes.length);
    }
    if (offset + length < 0) {
      throw new IllegalStateException(
          "offset+length is negative: offset=" + offset + ",length=" + length);
    }
    if (offset + length > bytes.length) {
      throw new IllegalStateException(
          "offset+length out of bounds: offset="
              + offset
              + ",length="
              + length
              + ",bytes.length="
              + bytes.length);
    }
    return true;
  }
}
