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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * A variety of high efficiency bit twiddling routines and encoders for primitives.
 */
public final class BitUtil {

  private BitUtil() {} // no instance

  /**
   * A {@link VarHandle} to read/write big endian {@code short} from a byte array. Shape: {@code
   * short vh.get(byte[] arr, int ofs)} and {@code void vh.set(byte[] arr, int ofs, short val)}
   *
   * @deprecated Better use little endian unless it is needed for backwards compatibility.
   */
  @Deprecated
  public static final VarHandle VH_BE_SHORT =
      MethodHandles.byteArrayViewVarHandle(short[].class, ByteOrder.BIG_ENDIAN);

  /**
   * A {@link VarHandle} to read/write big endian {@code int} from a byte array. Shape: {@code int
   * vh.get(byte[] arr, int ofs)} and {@code void vh.set(byte[] arr, int ofs, int val)}
   *
   * @deprecated Better use little endian unless it is needed for backwards compatibility.
   */
  @Deprecated
  public static final VarHandle VH_BE_INT =
      MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.BIG_ENDIAN);

  /**
   * A {@link VarHandle} to read/write big endian {@code long} from a byte array. Shape: {@code long
   * vh.get(byte[] arr, int ofs)} and {@code void vh.set(byte[] arr, int ofs, long val)}
   *
   * @deprecated Better use little endian unless it is needed for backwards compatibility.
   */
  @Deprecated
  public static final VarHandle VH_BE_LONG =
      MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.BIG_ENDIAN);

  /**
   * A {@link VarHandle} to read/write big endian {@code float} from a byte array. Shape: {@code
   * float vh.get(byte[] arr, int ofs)} and {@code void vh.set(byte[] arr, int ofs, float val)}
   *
   * @deprecated Better use little endian unless it is needed for backwards compatibility.
   */
  @Deprecated
  public static final VarHandle VH_BE_FLOAT =
      MethodHandles.byteArrayViewVarHandle(float[].class, ByteOrder.BIG_ENDIAN);

  /**
   * A {@link VarHandle} to read/write big endian {@code double} from a byte array. Shape: {@code
   * double vh.get(byte[] arr, int ofs)} and {@code void vh.set(byte[] arr, int ofs, double val)}
   *
   * @deprecated Better use little endian unless it is needed for backwards compatibility.
   */
  @Deprecated
  public static final VarHandle VH_BE_DOUBLE =
      MethodHandles.byteArrayViewVarHandle(double[].class, ByteOrder.BIG_ENDIAN);

  // magic numbers for bit interleaving
  private static final long MAGIC0 = 0x5555555555555555L;
  private static final long MAGIC1 = 0x3333333333333333L;
  private static final long MAGIC2 = 0x0F0F0F0F0F0F0F0FL;
  private static final long MAGIC3 = 0x00FF00FF00FF00FFL;
  private static final long MAGIC4 = 0x0000FFFF0000FFFFL;
  private static final long MAGIC5 = 0x00000000FFFFFFFFL;
  private static final long MAGIC6 = 0xAAAAAAAAAAAAAAAAL;

  // shift values for bit interleaving
  private static final long SHIFT0 = 1;
  private static final long SHIFT1 = 2;
  private static final long SHIFT2 = 4;
  private static final long SHIFT3 = 8;
  private static final long SHIFT4 = 16;
}
