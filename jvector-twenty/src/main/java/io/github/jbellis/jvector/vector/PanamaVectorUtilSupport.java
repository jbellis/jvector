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

import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;

final class
PanamaVectorUtilSupport implements VectorUtilSupport
{
    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, (ArrayVectorFloat)b);
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return SimdOps.cosineSimilarity((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, VectorFloat<?> b) {
        return SimdOps.squareDistance((ArrayVectorFloat)a, (ArrayVectorFloat)b);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.squareDistance((ArrayVectorFloat) a, aoffset, (ArrayVectorFloat) b, boffset, length);
    }

    @Override
    public int dotProduct(VectorByte<?> a, VectorByte<?> b) {
        return SimdOps.dotProduct((ArrayVectorByte)a, (ArrayVectorByte)b);
    }

    @Override
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, aoffset, (ArrayVectorFloat)b, boffset, length);
    }

    @Override
    public float cosine(VectorByte<?> a, VectorByte<?> b) {
        return SimdOps.cosineSimilarity((ArrayVectorByte)a, (ArrayVectorByte)b);
    }

    @Override
    public int squareDistance(VectorByte<?> a, VectorByte<?> b) {
        return SimdOps.squareDistance((ArrayVectorByte)a, (ArrayVectorByte)b);
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        return SimdOps.sum(vectors);
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return SimdOps.sum((ArrayVectorFloat) vector);
    }

    @Override
    public void divInPlace(VectorFloat<?> vector, float divisor) {
        SimdOps.divInPlace((ArrayVectorFloat)vector, divisor);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        SimdOps.addInPlace((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
        return SimdOps.sub((ArrayVectorFloat)lhs, (ArrayVectorFloat)rhs);
    }
}
