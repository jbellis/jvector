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

import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Interface for implementations of VectorUtil support.
 */
public interface VectorUtilSupport {

  /** Calculates the dot product of the given float arrays. */
  float dotProduct(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the dot product of float arrays of differing sizes, or a subset of the data */
  float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the cosine similarity between the two vectors. */
  float cosine(VectorFloat<?> v1, VectorFloat<?> v2);

  /** Returns the sum of squared differences of the two vectors. */
  float squareDistance(VectorFloat<?> a, VectorFloat<?> b);

  /** Calculates the sum of squared differences of float arrays of differing sizes, or a subset of the data */
  float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length);

  /** Returns the dot product computed over signed bytes. */
  int dotProduct(VectorByte<?> a, VectorByte<?> b);

  /** Returns the cosine similarity between the two byte vectors. */
  float cosine(VectorByte<?> a, VectorByte<?> b);

  /** Returns the sum of squared differences of the two byte vectors. */
  int squareDistance(VectorByte<?> a, VectorByte<?> b);


  /** returns the sum of the given vectors. */
  VectorFloat<?> sum(List<VectorFloat<?>> vectors);

  /** return the sum of the components of the vector */
  float sum(VectorFloat<?> vector);

  /** Divide vector by divisor, in place (vector will be modified) */
  void divInPlace(VectorFloat<?> vector, float divisor);

  /** Adds v2 into v1, in place (v1 will be modified) */
  void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2);

  /** @return lhs - rhs, element-wise */
  VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs);
}
