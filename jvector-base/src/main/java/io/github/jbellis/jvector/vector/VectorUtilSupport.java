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

package io.github.jbellis.jvector.vector;

import java.util.List;

/**
 * Interface for implementations of VectorUtil support.
 */
public interface VectorUtilSupport {

  /** Calculates the dot product of the given float arrays. */
  float dotProduct(float[] a, float[] b);

  /** Calculates the dot product of float arrays of differing sizes, or a subset of the data */
  float dotProduct(float[] a, int aoffset, float[] b, int boffset, int length);

  /** Returns the cosine similarity between the two vectors. */
  float cosine(float[] v1, float[] v2);

  /** Returns the sum of squared differences of the two vectors. */
  float squareDistance(float[] a, float[] b);

  /** Returns the dot product computed over signed bytes. */
  int dotProduct(byte[] a, byte[] b);

  /** Returns the cosine similarity between the two byte vectors. */
  float cosine(byte[] a, byte[] b);

  /** Returns the sum of squared differences of the two byte vectors. */
  int squareDistance(byte[] a, byte[] b);

  /** returns the sum of the given vectors. */
  float[] sum(List<float[]> vectors);

  /** return the sum of the components of the vector */
  float sum(float[] vector);

  /** Divide vector by divisor, in place (vector will be modified) */
  void divInPlace(float[] vector, float divisor);

  /** Adds v2 into v1, in place (v1 will be modified) */
  public void addInPlace(float[] v1, float[] v2);

  /** @return lhs - rhs, element-wise */
  public float[] sub(float[] lhs, float[] rhs);
}
