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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.util.DocIdSetIterator;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import org.junit.Before;

/** Tests KNN graphs */
public class TestByteVectorGraph extends GraphIndexTestCase<VectorByte<?>> {

  @Before
  public void setup() {
    similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
  }

  @Override
  VectorEncoding getVectorEncoding() {
    return VectorEncoding.BYTE;
  }

  @Override
  VectorByte<?> randomVector(int dim) {
    return randomVector8(getRandom(), dim);
  }

  @Override
  AbstractMockVectorValues<VectorByte<?>> vectorValues(int size, int dimension) {
    return MockByteVectorValues.fromValues(createRandomByteVectors(size, dimension, getRandom()));
  }

  static boolean fitsInByte(float v) {
    return v <= 127 && v >= -128 && v % 1 == 0;
  }

  @Override
  AbstractMockVectorValues<VectorByte<?>> vectorValues(VectorFloat<?>[] values) {
    VectorByte<?>[] bValues = new VectorByte<?>[values.length];
    // The case when all floats fit within a byte already.
    boolean scaleSimple = fitsInByte(values[0].get(0));
    for (int i = 0; i < values.length; i++) {
      bValues[i] = vectorTypeSupport.createByteType(values[i].length());
      for (int j = 0; j < values[i].length(); j++) {
        final float v;
        if (scaleSimple) {
          assert fitsInByte(values[i].get(j));
          v = values[i].get(j);
        } else {
          v = values[i].get(j) * 127;
        }
        bValues[i].set(j, (byte) v);
      }
    }
    return MockByteVectorValues.fromValues(bValues);
  }

  @Override
  AbstractMockVectorValues<VectorByte<?>> vectorValues(
      int size,
      int dimension,
      AbstractMockVectorValues<VectorByte<?>> pregeneratedVectorValues,
      int pregeneratedOffset) {
    VectorByte<?>[] vectors = new VectorByte<?>[size];
    VectorByte<?>[] randomVectors =
        createRandomByteVectors(size - pregeneratedVectorValues.values.length, dimension, getRandom());

    for (int i = 0; i < pregeneratedOffset; i++) {
      vectors[i] = randomVectors[i];
    }

    int currentDoc;
    while ((currentDoc = pregeneratedVectorValues.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
      vectors[pregeneratedOffset + currentDoc] = pregeneratedVectorValues.values[currentDoc];
    }

    for (int i = pregeneratedOffset + pregeneratedVectorValues.values.length;
        i < vectors.length;
        i++) {
      vectors[i] = randomVectors[i - pregeneratedVectorValues.values.length];
    }

    return MockByteVectorValues.fromValues(vectors);
  }

  @Override
  RandomAccessVectorValues<VectorByte<?>> circularVectorValues(int nDoc) {
    return new CircularByteVectorValues(nDoc);
  }

  @Override
  VectorByte<?> getTargetVector() {
    return vectorTypeSupport.createByteType(new byte[] {1, 0});
  }
}
