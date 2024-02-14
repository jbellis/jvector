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

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

/**
 * VectorUtilSupport implementation that prefers native/Panama SIMD.
 */
final class NativeVectorUtilSupport implements VectorUtilSupport
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
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return VectorSimdOps.dotProduct((OffHeapVectorFloat)a, aoffset, (OffHeapVectorFloat)b, boffset, length);
        //return NativeSimdOps.dot_product_f32(FloatVector.SPECIES_PREFERRED.vectorBitSize(), ((OffHeapVectorFloat)a).get(), aoffset, ((OffHeapVectorFloat)b).get(), boffset, length);
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
    public void scale(VectorFloat<?> vector, float multiplier) {
        VectorSimdOps.scale((OffHeapVectorFloat) vector, multiplier);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        VectorSimdOps.addInPlace((OffHeapVectorFloat)v1, (OffHeapVectorFloat)v2);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        VectorSimdOps.subInPlace((OffHeapVectorFloat)v1, (OffHeapVectorFloat)v2);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> lhs, VectorFloat<?> rhs) {
        return VectorSimdOps.sub((OffHeapVectorFloat)lhs, (OffHeapVectorFloat)rhs);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        return NativeSimdOps.assemble_and_sum_f32_512(((OffHeapVectorFloat)data).get(), dataBase, ((OffHeapByteSequence)baseOffsets).get(), baseOffsets.length());
    }

    @Override
    public int hammingDistance(long[] v1, long[] v2) {
        return VectorSimdOps.hammingDistance(v1, v2);
    }

    @Override
    public void bulkShuffleSimilarity(ByteSequence<?> shuffles, int codebookCount, VectorFloat<?> partials, VectorSimilarityFunction vsf, VectorFloat<?> results) {
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.bulk_shuffle_dot_f32_512(((OffHeapByteSequence) shuffles).get(), codebookCount, ((OffHeapVectorFloat) partials).get(), ((OffHeapVectorFloat) results).get());
            case EUCLIDEAN -> NativeSimdOps.bulk_shuffle_euclidean_f32_512(((OffHeapByteSequence) shuffles).get(), codebookCount, ((OffHeapVectorFloat) partials).get(), ((OffHeapVectorFloat) results).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for bulkShuffleSimilarity");
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookBase, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        switch (vsf) {
            case DOT_PRODUCT -> NativeSimdOps.calculate_partial_sums_dot_f32_512(((OffHeapVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((OffHeapVectorFloat)query).get(), queryOffset, ((OffHeapVectorFloat)partialSums).get());
            case EUCLIDEAN -> NativeSimdOps.calculate_partial_sums_euclidean_f32_512(((OffHeapVectorFloat)codebook).get(), codebookBase, size, clusterCount, ((OffHeapVectorFloat)query).get(), queryOffset, ((OffHeapVectorFloat)partialSums).get());
            case COSINE -> throw new UnsupportedOperationException("Cosine similarity not supported for calculatePartialSums");
        }
    }
}
