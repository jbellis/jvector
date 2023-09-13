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

package com.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.github.jbellis.jvector.vector.VectorEncoding;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Before;

import static com.github.jbellis.jvector.util.DocIdSetIterator.NO_MORE_DOCS;

/** Tests KNN graphs */
public class TestByteVectorGraph extends GraphIndexTestCase<byte[]> {

  @Before
  public void setup() {
    similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
  }

  @Override
  VectorEncoding getVectorEncoding() {
    return VectorEncoding.BYTE;
  }

  @Override
  byte[] randomVector(int dim) {
    return randomVector8(getRandom(), dim);
  }

  @Override
  AbstractMockVectorValues<byte[]> vectorValues(int size, int dimension) {
    return MockByteVectorValues.fromValues(createRandomByteVectors(size, dimension, getRandom()));
  }

  static boolean fitsInByte(float v) {
    return v <= 127 && v >= -128 && v % 1 == 0;
  }

  @Override
  AbstractMockVectorValues<byte[]> vectorValues(float[][] values) {
    byte[][] bValues = new byte[values.length][];
    // The case when all floats fit within a byte already.
    boolean scaleSimple = fitsInByte(values[0][0]);
    for (int i = 0; i < values.length; i++) {
      bValues[i] = new byte[values[i].length];
      for (int j = 0; j < values[i].length; j++) {
        final float v;
        if (scaleSimple) {
          assert fitsInByte(values[i][j]);
          v = values[i][j];
        } else {
          v = values[i][j] * 127;
        }
        bValues[i][j] = (byte) v;
      }
    }
    return MockByteVectorValues.fromValues(bValues);
  }

  @Override
  AbstractMockVectorValues<byte[]> vectorValues(
      int size,
      int dimension,
      AbstractMockVectorValues<byte[]> pregeneratedVectorValues,
      int pregeneratedOffset) {
    byte[][] vectors = new byte[size][];
    byte[][] randomVectors =
        createRandomByteVectors(size - pregeneratedVectorValues.values.length, dimension, getRandom());

    for (int i = 0; i < pregeneratedOffset; i++) {
      vectors[i] = randomVectors[i];
    }

    int currentDoc;
    while ((currentDoc = pregeneratedVectorValues.nextDoc()) != NO_MORE_DOCS) {
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
  RandomAccessVectorValues<byte[]> circularVectorValues(int nDoc) {
    return new CircularByteVectorValues(nDoc);
  }

  @Override
  byte[] getTargetVector() {
    return new byte[] {1, 0};
  }
}
