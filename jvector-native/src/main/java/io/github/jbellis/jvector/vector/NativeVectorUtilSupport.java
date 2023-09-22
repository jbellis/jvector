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

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.FloatVector;

final class
NativeVectorUtilSupport implements VectorUtilSupport
{
    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return this.dotProduct(a, 0, b, 0, a.length());
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return VectorSimdOps.cosineSimilarity((OffHeapVectorFloat)v1, (OffHeapVectorFloat)v2);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, VectorFloat<?> b) {
         return VectorSimdOps.squareDistance((OffHeapVectorFloat)a, (OffHeapVectorFloat)b);
    }

    @Override
    public float squareDistance(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return VectorSimdOps.squareDistance((OffHeapVectorFloat) a, aoffset, (OffHeapVectorFloat) b, boffset, length);
    }

    @Override
    public int dotProduct(VectorByte<?> a, VectorByte<?> b) {
        return VectorSimdOps.dotProduct((OffHeapVectorByte)a, (OffHeapVectorByte)b);
    }

    @Override
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return NativeSimdOps.dot_product_f32(FloatVector.SPECIES_PREFERRED.vectorBitSize(), ((OffHeapVectorFloat)a).get(), aoffset, ((OffHeapVectorFloat)b).get(), boffset, length);
    }

    @Override
    public float cosine(VectorByte<?> a, VectorByte<?> b) {
        return VectorSimdOps.cosineSimilarity((OffHeapVectorByte)a, (OffHeapVectorByte)b);
    }

    @Override
    public int squareDistance(VectorByte<?> a, VectorByte<?> b) {
        return VectorSimdOps.squareDistance((OffHeapVectorByte)a, (OffHeapVectorByte)b);
    }

    @Override
    public VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        return VectorSimdOps.sum(vectors);
    }

    @Override
    public float sum(VectorFloat<?> vector) {
        return VectorSimdOps.sum((OffHeapVectorFloat) vector);
    }

    @Override
    public void divInPlace(VectorFloat<?> vector, float divisor) {
        VectorSimdOps.divInPlace((OffHeapVectorFloat)vector, divisor);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        VectorSimdOps.addInPlace((OffHeapVectorFloat)v1, (OffHeapVectorFloat)v2);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
        return VectorSimdOps.sub((OffHeapVectorFloat)lhs, (OffHeapVectorFloat)rhs);
    }
}
