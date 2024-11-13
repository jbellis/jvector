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

import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.util.List;

final class PanamaVectorUtilSupport implements VectorUtilSupport {
    @Override
    public float dotProduct(VectorFloat<?> a, VectorFloat<?> b) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, (ArrayVectorFloat)b);
    }

    @Override
    public float cosine(VectorFloat<?> v1, VectorFloat<?> v2) {
        return SimdOps.cosineSimilarity((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public float cosine(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.cosineSimilarity((ArrayVectorFloat)a, aoffset, (ArrayVectorFloat)b, boffset, length);
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
    public float dotProduct(VectorFloat<?> a, int aoffset, VectorFloat<?> b, int boffset, int length) {
        return SimdOps.dotProduct((ArrayVectorFloat)a, aoffset, (ArrayVectorFloat)b, boffset, length);
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
    public void scale(VectorFloat<?> vector, float multiplier) {
        SimdOps.scale((ArrayVectorFloat) vector, multiplier);
    }

    @Override
    public void pow(VectorFloat<?> vector, float exponent) {
        SimdOps.pow((ArrayVectorFloat) vector, exponent);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        SimdOps.addInPlace((ArrayVectorFloat)v1, (ArrayVectorFloat)v2);
    }

    @Override
    public void addInPlace(VectorFloat<?> v1, float value) {
        SimdOps.addInPlace((ArrayVectorFloat)v1, value);
    }

    @Override
    public void subInPlace(VectorFloat<?> v1, VectorFloat<?> v2) {
        SimdOps.subInPlace((ArrayVectorFloat) v1, (ArrayVectorFloat) v2);
    }

    @Override
    public void subInPlace(VectorFloat<?> vector, float value) {
        SimdOps.subInPlace((ArrayVectorFloat) vector, value);
    }

    @Override
    public void constantMinusExponentiatedVector(VectorFloat<?> vector, float constant, float exponent) {
        SimdOps.constantMinusExponentiatedVector((ArrayVectorFloat) vector, constant, exponent);
    }

    @Override
    public void exponentiateConstantMinusVector(VectorFloat<?> vector, float constant, float exponent) {
        SimdOps.exponentiateConstantMinusVector((ArrayVectorFloat) vector, constant, exponent);
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, VectorFloat<?> b) {
        if (a.length() != b.length()) {
            throw new IllegalArgumentException("Vectors must be the same length");
        }
        return SimdOps.sub((ArrayVectorFloat)a, 0, (ArrayVectorFloat)b, 0, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, float value) {
        return SimdOps.sub((ArrayVectorFloat)a, 0, value, a.length());
    }

    @Override
    public VectorFloat<?> sub(VectorFloat<?> a, int aOffset, VectorFloat<?> b, int bOffset, int length) {
        return SimdOps.sub((ArrayVectorFloat) a, aOffset, (ArrayVectorFloat) b, bOffset, length);
    }

    @Override
    public float assembleAndSum(VectorFloat<?> data, int dataBase, ByteSequence<?> baseOffsets) {
        float sum = 0f;
        for (int i = 0; i < baseOffsets.length(); i++) {
            sum += data.get(dataBase * i + Byte.toUnsignedInt(baseOffsets.get(i)));
        }
        return sum;
    }

    @Override
    public int hammingDistance(long[] v1, long[] v2) {
        return SimdOps.hammingDistance(v1, v2);
    }

    @Override
    public float max(VectorFloat<?> vector) {
        return SimdOps.max((ArrayVectorFloat) vector);
    }

    @Override
    public float min(VectorFloat<?> vector) {
        return SimdOps.min((ArrayVectorFloat) vector);
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums) {
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    partialSums.set(codebookBase + i, dotProduct(codebook, i * size, query, queryOffset, size));
                    break;
                case EUCLIDEAN:
                    partialSums.set(codebookBase + i, squareDistance(codebook, i * size, query, queryOffset, size));
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
    }

    @Override
    public void calculatePartialSums(VectorFloat<?> codebook, int codebookIndex, int size, int clusterCount, VectorFloat<?> query, int queryOffset, VectorSimilarityFunction vsf, VectorFloat<?> partialSums, VectorFloat<?> partialBest) {
        float best = vsf == VectorSimilarityFunction.EUCLIDEAN ? Float.MAX_VALUE : -Float.MAX_VALUE;
        float val;
        int codebookBase = codebookIndex * clusterCount;
        for (int i = 0; i < clusterCount; i++) {
            switch (vsf) {
                case DOT_PRODUCT:
                    val = dotProduct(codebook, i * size, query, queryOffset, size);
                    partialSums.set(codebookBase + i, val);
                    best = Math.max(best, val);
                    break;
                case EUCLIDEAN:
                    val = squareDistance(codebook, i * size, query, queryOffset, size);
                    partialSums.set(codebookBase + i, val);
                    best = Math.min(best, val);
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported similarity function " + vsf);
            }
        }
        partialBest.set(codebookIndex, best);
    }

    @Override
    public void quantizePartials(float delta, VectorFloat<?> partials, VectorFloat<?> partialBases, ByteSequence<?> quantizedPartials) {
        SimdOps.quantizePartials(delta, (ArrayVectorFloat) partials, (ArrayVectorFloat) partialBases, (ArrayByteSequence) quantizedPartials);
    }

    @Override
    public float nvqDotProduct8bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias, float vectorSum) {
        return SimdOps.nvqDotProduct8bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions, a, b, scale, bias, vectorSum);
    }

    @Override
    public float nvqDotProduct4bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias, float vectorSum) {
        return SimdOps.nvqDotProduct4bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions, a, b, scale, bias, vectorSum);
    }

    @Override
    public float nvqSquareL2Distance8bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias) {
        return SimdOps.nvqSquareDistance8bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions, a, b, scale, bias);
    }

    @Override
    public float nvqSquareL2Distance4bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias) {
        return SimdOps.nvqSquareDistance4bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions, a, b, scale, bias);
    }

    @Override
    public float[] nvqCosine8bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias, VectorFloat<?> centroid) {
        return SimdOps.nvqCosine8bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions,  a, b, scale, bias,
                (ArrayVectorFloat) centroid
        );
    }

    @Override
    public float[] nvqCosine4bit(VectorFloat<?> vector, ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias, VectorFloat<?> centroid) {
        return SimdOps.nvqCosine4bit(
                (ArrayVectorFloat) vector, (ArrayByteSequence) bytes,
                originalDimensions,  a, b, scale, bias,
                (ArrayVectorFloat) centroid
        );
    }

    @Override
    public void nvqShuffleQueryInPlace4bit(VectorFloat<?> vector) {
        SimdOps.nvqShuffleQueryInPlace4bit((ArrayVectorFloat) vector);
    }

    @Override
    public VectorFloat<?> nvqDequantize8bit(ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias) {
        return SimdOps.nvqDequantize8bit((ArrayByteSequence) bytes, originalDimensions,  a, b, scale, bias);
    }

    @Override
    public VectorFloat<?> nvqDequantize4bit(ByteSequence<?> bytes, int originalDimensions, float a, float b, float scale, float bias) {
        return SimdOps.nvqDequantize4bit((ArrayByteSequence) bytes, originalDimensions,  a, b, scale, bias);
    }

    @Override
    public void nvqDequantize8bit(ByteSequence<?> bytes, float a, float b, float scale, float bias, VectorFloat<?> destination) {
        SimdOps.nvqDequantize8bit(
                (ArrayByteSequence) bytes,  a, b, scale, bias,
                (ArrayVectorFloat) destination
        );
    }

    @Override
    public void nvqDequantize4bit(ByteSequence<?> bytes, float a, float b, float scale, float bias, VectorFloat<?> destination) {
        SimdOps.nvqDequantize4bit(
                (ArrayByteSequence) bytes,  a, b, scale, bias,
                (ArrayVectorFloat) destination
        );
    }

    @Override
    public void nvqDequantizeUnnormalized8bit(ByteSequence<?> bytes, float a, float b, VectorFloat<?> destination) {
        SimdOps.nvqDequantizeUnnormalized8bit(
                (ArrayByteSequence) bytes, a, b,
                (ArrayVectorFloat) destination
        );
    }

    @Override
    public void nvqDequantizeUnnormalized4bit(ByteSequence<?> bytes, float a, float b, VectorFloat<?> destination) {
        SimdOps.nvqDequantizeUnnormalized4bit(
                (ArrayByteSequence) bytes, a, b,
                (ArrayVectorFloat) destination
        );
    }
}

