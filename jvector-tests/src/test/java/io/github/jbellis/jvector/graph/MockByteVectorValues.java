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

import io.github.jbellis.jvector.util.ArrayUtil;

class MockByteVectorValues extends AbstractMockVectorValues<byte[]> {
  private final byte[] denseScratch;

  static MockByteVectorValues fromValues(byte[][] values) {
    int dimension = values[0].length;
    int maxDoc = values.length;
    byte[][] denseValues = new byte[maxDoc][];
    int count = 0;
    for (int i = 0; i < maxDoc; i++) {
      if (values[i] != null) {
        denseValues[count++] = values[i];
      }
    }
    return new MockByteVectorValues(dimension, denseValues, count);
  }

  MockByteVectorValues(int dimension, byte[][] denseValues, int numVectors) {
    super(dimension, denseValues, numVectors);
    denseScratch = new byte[dimension];
  }

  @Override
  public MockByteVectorValues copy() {
    return new MockByteVectorValues(dimension,
                                    ArrayUtil.copyOfSubArray(denseValues, 0, denseValues.length),
                                    numVectors);
  }

  @Override
  public boolean isValueShared() {
    return true;
  }

  @Override
  public byte[] vectorValue(int targetOrd) {
    byte[] original = super.vectorValue(targetOrd);
    if (original == null) {
      return null;
    }
    // present a single vector reference to callers like the disk-backed RAVV implmentations,
    // to catch cases where they are not making a copy
    System.arraycopy(original, 0, denseScratch, 0, dimension);
    return denseScratch;
  }
}
