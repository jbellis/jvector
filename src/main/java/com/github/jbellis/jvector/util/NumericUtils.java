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

/*
 * This file has been modified by the JVector developers in compliance
 * with the Apache License, section 4.
 */

package com.github.jbellis.jvector.util;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * Helper APIs to encode numeric values as sortable bytes and vice-versa.
 *
 * <p>To also index floating point numbers, this class supplies two methods to convert them to
 * integer values by changing their bit layout: {@link #doubleToSortableLong}, {@link
 * #floatToSortableInt}. You will have no precision loss by converting floating point numbers to
 * integers and back (only that the integer form is not usable). Other data types like dates can
 * easily converted to longs or ints (e.g. date to long: {@link java.util.Date#getTime}).
 */
public final class NumericUtils {

  private NumericUtils() {} // no instance!

    /**
   * Converts a <code>float</code> value to a sortable signed <code>int</code>. The value is
   * converted by getting their IEEE 754 floating-point &quot;float format&quot; bit layout and then
   * some bits are swapped, to be able to compare the result as int. By this the precision is not
   * reduced, but the value can easily used as an int. The sort order (including {@link Float#NaN})
   * is defined by {@link Float#compareTo}; {@code NaN} is greater than positive infinity.
   *
   * @see #sortableIntToFloat
   */
  public static int floatToSortableInt(float value) {
    return sortableFloatBits(Float.floatToIntBits(value));
  }

  /**
   * Converts a sortable <code>int</code> back to a <code>float</code>.
   *
   * @see #floatToSortableInt
   */
  public static float sortableIntToFloat(int encoded) {
    return Float.intBitsToFloat(sortableFloatBits(encoded));
  }

  /** Converts IEEE 754 representation of a double to sortable order (or back to the original) */
  public static long sortableDoubleBits(long bits) {
    return bits ^ (bits >> 63) & 0x7fffffffffffffffL;
  }

  /** Converts IEEE 754 representation of a float to sortable order (or back to the original) */
  public static int sortableFloatBits(int bits) {
    return bits ^ (bits >> 31) & 0x7fffffff;
  }

    /**
   * Decodes a BigInteger value previously written with {@link #bigIntToSortableBytes}
   *
   * @see #bigIntToSortableBytes(BigInteger, int, byte[], int)
   */
  public static BigInteger sortableBytesToBigInt(byte[] encoded, int offset, int length) {
    byte[] bigIntBytes = new byte[length];
    System.arraycopy(encoded, offset, bigIntBytes, 0, length);
    // Flip the sign bit back to the original
    bigIntBytes[0] ^= 0x80;
    return new BigInteger(bigIntBytes);
  }
}
