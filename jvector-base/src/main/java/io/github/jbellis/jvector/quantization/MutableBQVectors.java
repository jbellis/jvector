/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.quantization;

public class MutableBQVectors extends BQVectors implements MutableCompressedVectors<long[]> {
    /**
     * Construct a mutable BQVectors instance with the given BinaryQuantization and maximum number of vectors
     * that will be stored in this instance.
     * @param bq the BinaryQuantization to use
     * @param maximumVectorCount the maximum number of vectors that will be stored in this instance
     */
    public MutableBQVectors(BinaryQuantization bq, int maximumVectorCount) {
        super(bq);
        this.compressedVectors = new long[maximumVectorCount][];
        this.vectorCount = 0;
    }

    @Override
    public void encodeAndSet(int ordinal, long[] vector) {
        compressedVectors[ordinal] = vector;
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }

    @Override
    public void setZero(int ordinal) {
        compressedVectors[ordinal] = new long[bq.compressedVectorSize()];
        vectorCount = Math.max(vectorCount, ordinal + 1);
    }
}
