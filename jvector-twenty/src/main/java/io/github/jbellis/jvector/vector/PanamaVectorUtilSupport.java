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

package io.github.jbellis.jvector.vector;

import java.util.List;

final class
PanamaVectorUtilSupport implements VectorUtilSupport
{

    @Override
    public float dotProduct(float[] a, float[] b) {
        return SimdOps.dotProduct(a, b);
    }

    @Override
    public float cosine(float[] v1, float[] v2) {
        return SimdOps.cosineSimilarity(v1, v2);
    }

    @Override
    public float squareDistance(float[] a, float[] b) {
        return SimdOps.squareDistance(a, b);
    }

    @Override
    public int dotProduct(byte[] a, byte[] b) {
        return SimdOps.dotProduct(a, b);
    }

    @Override
    public float dotProduct(float[] a, int aoffset, float[] b, int boffset, int length) {
        return SimdOps.dotProduct(a, aoffset, b, boffset, length);
    }

    @Override
    public float cosine(byte[] a, byte[] b) {
        return SimdOps.cosineSimilarity(a, b);
    }

    @Override
    public int squareDistance(byte[] a, byte[] b) {
        return SimdOps.squareDistance(a, b);
    }

    @Override
    public float[] sum(List<float[]> vectors) {
        return SimdOps.sum(vectors);
    }

    @Override
    public float sum(float[] vector) {
        return SimdOps.sum(vector);
    }

    @Override
    public void divInPlace(float[] vector, float divisor) {
        SimdOps.divInPlace(vector, divisor);
    }

    @Override
    public void addInPlace(float[] v1, float[] v2) {
        SimdOps.addInPlace(v1, v2);
    }

    @Override
    public float[] sub(float[] lhs, float[] rhs) {
        return SimdOps.sub(lhs, rhs);
    }
}
