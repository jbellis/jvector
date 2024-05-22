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

import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

/** Utilities for computations with numeric arrays */
public final class VectorUtil {

  private static final VectorUtilSupport impl =
      VectorizationProvider.getInstance().getVectorUtilSupport();

  private VectorUtil() {}

  /**
   * Returns the vector dot product of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.dotProduct(a, b);
    assert Float.isFinite(r) : String.format("dotProduct(%s, %s) = %s", a, b, r);
    return r;
  }

  public static float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
    //This check impacts FLOPS
    /*if ( length > Math.min(a.length - aoffset, b.length - boffset) ) {
      throw new IllegalArgumentException("length must be less than the vectors remaining space at the given offsets: a(" +
              (a.length - aoffset) + "), b(" + (b.length - boffset) + "), length(" + length + ")");
    }*/
    float r = impl.dotProduct(a, aoffset, b, boffset, length);
    assert Float.isFinite(r) : String.format("dotProduct(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the cosine similarity between the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float cosine(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.cosine(a, b);
    assert Float.isFinite(r) : String.format("cosine(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the sum of squared differences of the two vectors.
   *
   * @throws IllegalArgumentException if the vectors' dimensions differ.
   */
  public static float squareL2Distance(VectorFloat<?> a, VectorFloat<?> b) {
    if (a.length() != b.length()) {
      throw new IllegalArgumentException("vector dimensions differ: " + a.length() + "!=" + b.length());
    }
    float r = impl.squareDistance(a, b);
    assert Float.isFinite(r) : String.format("squareDistance(%s, %s) = %s", a, b, r);
    return r;
  }

  /**
   * Returns the sum of squared differences of the two vectors, or subvectors, of the given length.
   */
  public static float squareL2Distance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
    float r = impl.squareDistance(a, aoffset, b, boffset, length);
    assert Float.isFinite(r);
    return r;
  }

  /**
   * Modifies the argument to be unit length, dividing by its l2-norm. IllegalArgumentException is
   * thrown for zero vectors.
   *
   * @param v the vector to normalize
   */
  public static void l2normalize(VectorFloat<?> v) {
    double squareSum = dotProduct(v, v);
    if (squareSum == 0) {
      throw new IllegalArgumentException("Cannot normalize a zero-length vector");
    }
    double length = Math.sqrt(squareSum);
    scale(v, (float) (1.0 / length));
  }

  public static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
    if (vectors.isEmpty()) {
      throw new IllegalArgumentException("Input list cannot be empty");
    }

    return impl.sum(vectors);
  }

  public static float sum(VectorFloat<?> vector) {
    return impl.sum(vector);
  }

  public static void scale(VectorFloat<?> vector, float multiplier) {
    impl.scale(vector, multiplier);
  }

  public static void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.addInPlace(v1, v2);
  }

  public static void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
    impl.subInPlace(v1, v2);
  }

  public static VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
    return impl.sub(lhs, rhs);
  }

  public static VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
    return impl.sub(a, aOffset, b, bOffset, length);
  }

  public static float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> dataOffsets) {
    return impl.assembleAndSum(data, dataBase, dataOffsets);
  }

  public static void bulkShuffleQuantizedSimilarity(ByteSequence<?> shuffles, int codebookCount, ByteSequence<?> quantizedPartials, float delta, float minDistance, VectorFloat<?> results, VectorSimilarityFunction vsf) {
    impl.bulkShuffleQuantizedSimilarity(shuffles, codebookCount, quantizedPartials, delta, minDistance, vsf, results);
  }

  public static void bulkShuffleQuantizedSimilarityCosine(ByteSequence<?> shuffles, int codebookCount,
                                                          ByteSequence<?> quantizedPartialSums, float sumDelta, float minDistance,
                                                          ByteSequence<?> quantizedPartialMagnitudes, float magnitudeDelta, float minMagnitude,
                                                          float queryMagnitudeSquared, VectorFloat<?> results) {
    impl.bulkShuffleQuantizedSimilarityCosine(shuffles, codebookCount, quantizedPartialSums, sumDelta, minDistance, quantizedPartialMagnitudes, magnitudeDelta, minMagnitude, queryMagnitudeSquared, results);
  }

  public static int hammingDistance(long[] v1, long[] v2) {
    return impl.hammingDistance(v1, v2);
  }

  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBestDistances) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums, partialBestDistances);
  }

  public static void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int offset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
    impl.calculatePartialSums(codebook, codebookIndex, size, clusterCount, query, offset, vsf, partialSums);
  }

  public static void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBase, ByteSequence<?> quantizedPartials) {
    impl.quantizePartials(delta, partials, partialBase, quantizedPartials);
  }

  /**
   * Calculates the dot product similarity scores between v1 and multiple vectors packed into v2.
   * Note that unlike the dotProduct, this method puts similarity scores into results, taking this responsibility from VectorSimilarityFunction.
   * @param v1 the query vector
   * @param v2 multiple vectors to compare against
   * @param results the output vector to store the similarity scores. This should be pre-allocated to the same size as the number of vectors in v2.
   */
  public static void dotProductMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    impl.dotProductMultiScore(v1, v2, results);
  }

  /**
   * Calculates the Euclidean similarity scores between v1 and multiple vectors packed into v2.
   * Note that unlike the squareDistance, this method puts similarity scores into results, taking this responsibility from VectorSimilarityFunction.
   * @param v1 the query vector
   * @param v2 multiple vectors to compare against
   * @param results the output vector to store the similarity scores. This should be pre-allocated to the same size as the number of vectors in v2.
   */
  public static void euclideanMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    impl.squareL2DistanceMultiScore(v1, v2, results);
  }

  /**
   * Calculates the cosine similarity scores between v1 and multiple vectors packed into v2.
   * Note that unlike the cosine, this method puts similarity scores into results, taking this responsibility from VectorSimilarityFunction.
   * @param v1 the query vector
   * @param v2 multiple vectors to compare against
   * @param results the output vector to store the similarity scores. This should be pre-allocated to the same size as the number of vectors in v2.
   */
  public static void cosineMultiScore(VectorFloat<?> v1, VectorFloat<?> v2, VectorFloat<?> results) {
    impl.cosineMultiScore(v1, v2, results);
  }

  /**
   * Calculates the maximum value in the vector.
   * @param v vector
   * @return the maximum value, or -Float.MAX_VALUE if the vector is empty
   */
  public static float max(VectorFloat<?> v) {
    return impl.max(v);
  }

  /**
   * Calculates the minimum value in the vector.
   * @param v vector
   * @return the minimum value, or Float.MAX_VALUE if the vector is empty
   */
  public static float min(VectorFloat<?> v) {
    return impl.min(v);
  }

  /**
   * Calculates the dot product between the vector v and the LVQ-compressed vector quantizedVector.
   * @param v the uncompressed vector
   * @param quantizedVector the LVQ-compressed vector
   * @param querySum the horizontal sum of v
   * @return the dot product
   */
  public static float lvqDotProduct(VectorFloat<?> v, LocallyAdaptiveVectorQuantization.PackedVector quantizedVector, float querySum) {
    return impl.lvqDotProduct(v, quantizedVector, querySum);
  }

  /**
   * Calculates the square of the L2 distance between the centered vector centeredVector and the LVQ-compressed vector quantizedVector.
   * @param centeredV the de-meaned vector (using the same centroid as used to quantize quantizedVector)
   * @param quantizedVector the LVQ-compressed vector (using the same centroid as used to de-mean centeredV)
   * @return the square of the L2 distance
   */
  public static float lvqSquareL2Distance(VectorFloat<?> centeredV, LocallyAdaptiveVectorQuantization.PackedVector quantizedVector) {
    return impl.lvqSquareL2Distance(centeredV, quantizedVector);
  }

  /**
   * Calculates the cosine similarity between the vector v and LVQ-compressed vector quantizedVector.
   * @param v the uncompressed vector
   * @param quantizedVector the LVQ-compressed vector
   * @param centroid the centroid used to de-mean quantizedVector
   * @return the cosine similarity
   */
  public static float lvqCosine(VectorFloat<?> v, LocallyAdaptiveVectorQuantization.PackedVector quantizedVector, VectorFloat<?> centroid) {
    return impl.lvqCosine(v, quantizedVector, centroid);
  }
}
