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

import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.*;

import java.util.List;

final class SimdOps {

    static final boolean HAS_AVX512 = IntVector.SPECIES_PREFERRED == IntVector.SPECIES_512;
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    static float sum(ArrayVectorFloat vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            sum = sum.add(a);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            res += vector.get(i);
        }

        return res;
    }

    static VectorFloat<?> sum(List<VectorFloat<?>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length();
        ArrayVectorFloat sum = new ArrayVectorFloat(dimension);

        // Process each vector from the list
        for (VectorFloat<?> vector : vectors) {
            addInPlace(sum, (ArrayVectorFloat) vector);
        }

        return sum;
    }

    static void scale(ArrayVectorFloat vector, float multiplier) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var divResult = a.mul(multiplier);
            divResult.intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) * multiplier);
        }
    }

    static void pow(ArrayVectorFloat vector, float exponent) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            a.pow(exponent).intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, (float) Math.pow(vector.get(i), exponent));
        }
    }

    static float dot64(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dot128(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_128, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_128, v2.get(), offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dot256(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_256, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_256, v2.get(), offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dotPreferred(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), offset2);
        return a.mul(b).reduceLanes(VectorOperators.ADD);
    }

    static float dotProduct(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        return dotProduct(v1, 0, v2, 0, v1.length());
    }

    static float dotProduct(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return dotProductPreferred(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return dotProduct64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return dotProduct128(v1, v1offset, v2, v2offset, length);
        else
            return dotProduct256(v1, v1offset, v2, v2offset, length);

    }

    static float dotProduct64(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_64.length())
            return dot64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);
        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float dotProduct128(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_128.length())
            return dot128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_128, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_128, v2.get(), v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }


    static float dotProduct256(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_256.length())
            return dot256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_256, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_256, v2.get(), v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float dotProductPreferred(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_PREFERRED.length())
            return dotPreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), v2offset + i);
            sum = a.fma(b, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static float cosineSimilarity(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            sum += v1.get(i) * v2.get(i);
            aMagnitude += v1.get(i) * v1.get(i);
            bMagnitude += v2.get(i) * v2.get(i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    static float cosineSimilarity(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {
        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), v1offset + i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), v2offset + i);
            vsum = a.fma(b, vsum);
            vaMagnitude = a.fma(a, vaMagnitude);
            vbMagnitude = b.fma(b, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        for (int i = vectorizedLength; i < length; i++) {
            sum += v1.get(v1offset + i) * v2.get(v2offset + i);
            aMagnitude += v1.get(v1offset + i) * v1.get(v1offset + i);
            bMagnitude += v2.get(v2offset + i) * v2.get(v2offset + i);
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    static float squareDistance64(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance128(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_128, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_128, v2.get(), offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance256(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_256, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_256, v2.get(), offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistancePreferred(ArrayVectorFloat v1, int offset1, ArrayVectorFloat v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), offset2);
        var diff = a.sub(b);
        return diff.mul(diff).reduceLanes(VectorOperators.ADD);
    }

    static float squareDistance(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        return squareDistance(v1, 0, v2, 0, v1.length());
    }

    static float squareDistance(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, final int length)
    {
        //Common case first
        if (length >= FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset, length);

        if (length < FloatVector.SPECIES_128.length())
            return squareDistance64(v1, v1offset, v2, v2offset, length);
        else if (length < FloatVector.SPECIES_256.length())
            return squareDistance128(v1, v1offset, v2, v2offset, length);
        else
            return squareDistance256(v1, v1offset, v2, v2offset, length);
    }

    static float squareDistance64(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_64.length())
            return squareDistance64(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_64.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_64);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_64.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    static float squareDistance128(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_128.length())
            return squareDistance128(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_128.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_128);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_128.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_128, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_128, v2.get(), v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }


    static float squareDistance256(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {
        if (length == FloatVector.SPECIES_256.length())
            return squareDistance256(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_256, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_256, v2.get(), v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    static float squareDistancePreferred(ArrayVectorFloat v1, int v1offset, ArrayVectorFloat v2, int v2offset, int length) {

        if (length == FloatVector.SPECIES_PREFERRED.length())
            return squareDistancePreferred(v1, v1offset, v2, v2offset);

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int i = 0;
        // Process the vectorized part
        for (; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), v1offset + i);
            FloatVector b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), v2offset + i);
            var diff = a.sub(b);
            sum = diff.fma(diff, sum);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    static void addInPlace64(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), 0);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), 0);
        a.add(b).intoArray(v1.get(), 0);
    }

    static void addInPlace64(ArrayVectorFloat v1, float value) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), 0);
        a.add(value).intoArray(v1.get(), 0);
    }

    static void addInPlace(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        if (v1.length() == 2) {
            addInPlace64(v1, v2);
            return;
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), i);
            a.add(b).intoArray(v1.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i,  v1.get(i) + v2.get(i));
        }
    }

    static void addInPlace(ArrayVectorFloat v1, float value) {
        if (v1.length() == 2) {
            addInPlace64(v1, value);
            return;
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), i);
            a.add(value).intoArray(v1.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i,  v1.get(i) + value);
        }
    }

    static void subInPlace(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1.get(), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2.get(), i);
            a.sub(b).intoArray(v1.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            v1.set(i,  v1.get(i) - v2.get(i));
        }
    }

    static void subInPlace(ArrayVectorFloat vector, float value) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            a.sub(value).intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i,  vector.get(i) - value);
        }
    }

    static void constantMinusExponentiatedVector(ArrayVectorFloat vector, float constant, float exponent) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var subResult = a.pow(exponent).neg().add(constant);
            subResult.intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, constant - (float) Math.pow(vector.get(i), exponent));
        }
    }

    static void exponentiateConstantMinusVector(ArrayVectorFloat vector, float constant, float exponent) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var subResult = a.neg().add(constant).pow(exponent);
            subResult.intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, (float) Math.pow(constant - vector.get(i), exponent));
        }
    }

    static VectorFloat<?> sub(ArrayVectorFloat a, int aOffset, ArrayVectorFloat b, int bOffset, int length) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        float[] res = new float[length];

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a.get(), aOffset + i);
            var rhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, b.get(), bOffset + i);
            var subResult = lhs.sub(rhs);
            subResult.intoArray(res, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res[i] = a.get(aOffset + i) - b.get(bOffset + i);
        }

        return new ArrayVectorFloat(res);
    }

    static VectorFloat<?> sub(ArrayVectorFloat a, int aOffset, float value, int length) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(length);
        float[] res = new float[length];

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var lhs = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a.get(), aOffset + i);
            var subResult = lhs.sub(value);
            subResult.intoArray(res, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < length; i++) {
            res[i] = a.get(aOffset + i) - value;
        }

        return new ArrayVectorFloat(res);
    }


    static float assembleAndSum(float[] data, int dataBase, byte[] baseOffsets) {
        return HAS_AVX512 ? assembleAndSum512(data, dataBase, baseOffsets)
               : assembleAndSum256(data, dataBase, baseOffsets);
    }

    static float assembleAndSum512(float[] data, int dataBase, byte[] baseOffsets) {
        int[] convOffsets = scratchInt512.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_512);
        int i = 0;
        int limit = ByteVector.SPECIES_128.loopBound(baseOffsets.length);

        for (; i < limit; i += ByteVector.SPECIES_128.length()) {
            var scale = IntVector.zero(IntVector.SPECIES_512).addIndex(1).add(i).mul(dataBase);

            ByteVector.fromArray(ByteVector.SPECIES_128, baseOffsets, i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_512)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets,0);

            sum = sum.add(FloatVector.fromArray(FloatVector.SPECIES_512, data, 0, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        //Process tail
        for (; i < baseOffsets.length; i++)
            res += data[dataBase * i + Byte.toUnsignedInt(baseOffsets[i])];

        return res;
    }

    static float assembleAndSum256(float[] data, int dataBase, byte[] baseOffsets) {
        int[] convOffsets = scratchInt256.get();
        FloatVector sum = FloatVector.zero(FloatVector.SPECIES_256);
        int i = 0;
        int limit = ByteVector.SPECIES_64.loopBound(baseOffsets.length);

        for (; i < limit; i += ByteVector.SPECIES_64.length()) {
            var scale = IntVector.zero(IntVector.SPECIES_256).addIndex(1).add(i).mul(dataBase);

            ByteVector.fromArray(ByteVector.SPECIES_64, baseOffsets, i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, BYTE_TO_INT_MASK_256)
                    .reinterpretAsInts()
                    .add(scale)
                    .intoArray(convOffsets,0);

            sum = sum.add(FloatVector.fromArray(FloatVector.SPECIES_256, data, 0, convOffsets, 0));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process tail
        for (; i < baseOffsets.length; i++)
            res += data[dataBase * i + Byte.toUnsignedInt(baseOffsets[i])];

        return res;
    }

    /**
     * Vectorized calculation of Hamming distance for two arrays of long integers.
     * Both arrays should have the same length.
     *
     * @param a The first array
     * @param b The second array
     * @return The Hamming distance
     */
    public static int hammingDistance(long[] a, long[] b) {
        var sum = LongVector.zero(LongVector.SPECIES_PREFERRED);
        int vectorizedLength = LongVector.SPECIES_PREFERRED.loopBound(a.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += LongVector.SPECIES_PREFERRED.length()) {
            var va = LongVector.fromArray(LongVector.SPECIES_PREFERRED, a, i);
            var vb = LongVector.fromArray(LongVector.SPECIES_PREFERRED, b, i);

            var xorResult = va.lanewise(VectorOperators.XOR, vb);
            sum = sum.add(xorResult.lanewise(VectorOperators.BIT_COUNT));
        }

        int res = (int) sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < a.length; i++) {
            res += Long.bitCount(a[i] ^ b[i]);
        }

        return res;
    }

    public static float max(ArrayVectorFloat v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v.get(), i);
            accum = accum.max(a);
        }
        float max = accum.reduceLanes(VectorOperators.MAX);
        for (int i = vectorizedLength; i < v.length(); i++) {
            max = Math.max(max, v.get(i));
        }
        return max;
    }

    public static float min(ArrayVectorFloat v) {
        var accum = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, Float.MAX_VALUE);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v.get(), i);
            accum = accum.min(a);
        }
        float min = accum.reduceLanes(VectorOperators.MIN);
        for (int i = vectorizedLength; i < v.length(); i++) {
            min = Math.min(min, v.get(i));
        }
        return min;
    }

    public static void quantizePartials(float delta, ArrayVectorFloat partials, ArrayVectorFloat partialBases, ArrayByteSequence quantizedPartials) {
        var codebookSize = partials.length() / partialBases.length();
        var codebookCount = partialBases.length();

        for (int i = 0; i < codebookCount; i++) {
            var vectorizedLength = FloatVector.SPECIES_512.loopBound(codebookSize);
            var codebookBase = partialBases.get(i);
            var codebookBaseVector = FloatVector.broadcast(FloatVector.SPECIES_512, codebookBase);
            int j = 0;
            for (; j < vectorizedLength; j += FloatVector.SPECIES_512.length()) {
                var partialVector = FloatVector.fromArray(FloatVector.SPECIES_512, partials.get(), i * codebookSize + j);
                var quantized = (partialVector.sub(codebookBaseVector)).div(delta);
                quantized = quantized.max(FloatVector.zero(FloatVector.SPECIES_512)).min(FloatVector.broadcast(FloatVector.SPECIES_512, 65535));
                var quantizedBytes = (ShortVector) quantized.convertShape(VectorOperators.F2S, ShortVector.SPECIES_256, 0);
                quantizedBytes.reinterpretAsBytes().intoArray(quantizedPartials.get(), 2 * (i * codebookSize + j));
            }
            for (; j < codebookSize; j++) {
                var val = partials.get(i * codebookSize + j);
                var quantized = (short) Math.min((val - codebookBase) / delta, 65535);
                quantizedPartials.setLittleEndianShort(i * codebookSize + j, quantized);
            }
        }
    }

    //---------------------------------------------
    // NVQ quantization instructions start here
    //---------------------------------------------

    static FloatVector inverseKumaraswamy(FloatVector vector, float a, float b) {
        var res = vector.neg().add(1.f).pow(1.f / b);   // 1 - v ** (1 / a)
        res = res.neg().add(1.f).pow(1.f / a);          // 1 - v ** (1 / b)
        return res;
    }

    static float inverseKumaraswamy(float value, float a, float b) {
        var temp = (float) Math.pow(1 - value, 1.f / b);
        return (float) Math.pow(1 - temp, 1.f / a);
    }


    static void nvqDequantizeUnnormalized4bit(ArrayByteSequence bytes, float a, float b, ArrayVectorFloat res) {
        /*
         * bytes:      |  0   |  1   |  2   |  3   |  4   |  5     |  6     |  7     |
         * half-bytes: | 0  1 | 2  3 | 4  5 | 6  7 | 8  9 | 10  11 | 12  13 | 14  15 |
         *
         * 1st pass of original array:
         * & 0xf:      | 0    | 2    | 4    | 6    | 8    | 10     | 12     | 14     |
         * 2nd pass of original array:
         * lS 4:       | 1    | 3    | 5    | 7    | 9    | 11     | 13     | 15     |
         */
        var resArr = res.get();

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(bytes.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var arr = ByteVector.fromArray(ByteVector.SPECIES_64, bytes.get(), i);
            // 1st pass
            var subResult = arr.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, 0xf)
                    .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0)
                    .reinterpretAsFloats()
                    .div(15.f);

            subResult = inverseKumaraswamy(subResult, a, b);
            subResult.intoArray(resArr, 2 * i);

            // 2nd pass
            subResult = arr.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, 0xff)
                    .lanewise(VectorOperators.LSHR, 4)
                    .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0)
                    .reinterpretAsFloats()
                    .div(15.f);

            subResult = inverseKumaraswamy(subResult, a, b);
            subResult.intoArray(resArr, 2 * i + ByteVector.SPECIES_64.length());
        }

        // Process the tail
        if (vectorizedLength < bytes.length()) {
            for (int i = vectorizedLength; i < bytes.length() - 1; i++) {
                var intValue = Byte.toUnsignedInt(bytes.get(i));
                resArr[2 * i] = inverseKumaraswamy(intValue & 0xf, a, b);
                resArr[2 * i + 1] = inverseKumaraswamy(intValue << 4, a, b);
            }

            vectorizedLength = bytes.length() - 1;
            var intValue = Byte.toUnsignedInt(bytes.get(vectorizedLength));
            resArr[2 * vectorizedLength] = inverseKumaraswamy(intValue & 0xf, a, b);
            if (vectorizedLength * 2 == resArr.length) {
                resArr[2 * vectorizedLength + 1] = inverseKumaraswamy(intValue << 4, a, b);
            }
        }
    }

    static void nvqDequantizeUnnormalized8bit(ArrayByteSequence bytes, float a, float b, ArrayVectorFloat res) {
        var resArr = res.get();

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(bytes.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var arr = ByteVector.fromArray(ByteVector.SPECIES_64, bytes.get(), i)
                    .convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.AND, 0xff)
                    .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0)
                    .reinterpretAsFloats()
                    .div(255.f);

            arr = inverseKumaraswamy(arr, a, b);
            arr.intoArray(resArr, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < bytes.length(); i++) {
            resArr[i] = inverseKumaraswamy(bytes.get(i), a, b);
        }
    }

    static ArrayVectorFloat nvqDequantize8bit(ArrayByteSequence bytes, int originalDimensions, float a, float b, float scale, float bias) {
        var res = new ArrayVectorFloat(new float[originalDimensions]);
        nvqDequantize8bit(bytes, a, b, scale, bias, res);
        return res;
    }

    static ArrayVectorFloat nvqDequantize4bit(ArrayByteSequence bytes, int originalDimensions, float a, float b, float scale, float bias) {
        var res = new ArrayVectorFloat(new float[originalDimensions]);
        nvqDequantize4bit(bytes, a, b, scale, bias, res);
        return res;
    }

    static void nvqDequantize8bit(ArrayByteSequence bytes, float a, float b, float scale, float bias, ArrayVectorFloat destination) {
        nvqDequantizeUnnormalized8bit(bytes, a, b, destination);
        scale(destination, scale);
        addInPlace(destination, bias);
    }

    static void nvqDequantize4bit(ArrayByteSequence bytes, float a, float b, float scale, float bias, ArrayVectorFloat destination) {
        nvqDequantizeUnnormalized4bit(bytes, a, b, destination);
        scale(destination, scale);
        addInPlace(destination, bias);
    }

    /**
     * Compute the squared L2 distance for 8-bit NVQ
     * Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
     * first de-meaned by subtracting the global mean.
     *
     * The squared L2 distance is calculated between the query and quantized sub-vectors as follows:
     *
     * |query - vector|^2 \approx |query - scale * quantized + bias + globalMean|^2
     *                          = |(query - globalMean) - scale * quantized + bias|^2
     *
     * @param vector The shifted query (precomputed query - globalMean)
     * @param quantizedVector A quantized vector from the index
     * @return The square L2 distance
     */
    static float nvqSquareDistance8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias) {
        ArrayVectorFloat dequantizedVector = nvqDequantize8bit(quantizedVector, originalDimensions, a, b, scale, bias);

        // Assumes global mean removed from vector
        return squareDistance(vector, dequantizedVector);
    }

    /**
     * Compute the squared L2 distance for 4-bit NVQ
     * Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
     * first de-meaned by subtracting the global mean.
     *
     * The squared L2 distance is calculated between the query and quantized sub-vectors as follows:
     *
     * |query - vector|^2 \approx |query - scale * quantized + bias + globalMean|^2
     *                          = |(query - globalMean) - scale * quantized + bias|^2
     *
     * @param vector The shifted query (precomputed query - globalMean)
     * @param quantizedVector A quantized vector from the index
     * @return The square L2 distance
     */
    static float nvqSquareDistance4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias) {
        ArrayVectorFloat dequantizedVector = nvqDequantize4bit(quantizedVector, originalDimensions, a, b, scale, bias);

        // Assumes global mean removed from vector
        return squareDistance(vector, dequantizedVector);
    }

    static float nvqDotProduct8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias, float vectorSum) {
        ArrayVectorFloat dequantizedVector = new ArrayVectorFloat(new float[originalDimensions]);
        nvqDequantizeUnnormalized8bit(quantizedVector, a, b, dequantizedVector);

        float dotProd = dotProduct(vector, dequantizedVector);
        return scale * dotProd + bias * vectorSum;
    }

    static float nvqDotProduct4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias, float vectorSum) {
        ArrayVectorFloat dequantizedVector = new ArrayVectorFloat(new float[originalDimensions]);
        nvqDequantizeUnnormalized4bit(quantizedVector, a, b, dequantizedVector);

        float dotProd = dotProduct(vector, dequantizedVector);
        return scale * dotProd + bias * vectorSum;
    }

    static float[] nvqCosine8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias, ArrayVectorFloat centroid) {
        if ((vector.length() != originalDimensions) && (originalDimensions != centroid.length())) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        ArrayVectorFloat dequantizedVector = nvqDequantize8bit(quantizedVector, originalDimensions, a, b, scale, bias);
        addInPlace(dequantizedVector, centroid);

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var vb = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, dequantizedVector.get(), i);
            vsum = va.fma(vb, vsum);
            vbMagnitude = vb.fma(vb, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            sum += vector.get(i) * dequantizedVector.get(i);
            bMagnitude += dequantizedVector.get(i) * dequantizedVector.get(i);
        }

        return new float[]{sum, bMagnitude};
    }

    static float[] nvqCosine4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, int originalDimensions, float a, float b, float scale, float bias, ArrayVectorFloat centroid) {
        if ((vector.length() != originalDimensions) && (originalDimensions != centroid.length())) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        ArrayVectorFloat dequantizedVector = nvqDequantize4bit(quantizedVector, originalDimensions, a, b, scale, bias);
        addInPlace(dequantizedVector, centroid);

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var vb = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, dequantizedVector.get(), i);
            vsum = va.fma(vb, vsum);
            vbMagnitude = vb.fma(vb, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            sum += vector.get(i) * dequantizedVector.get(i);
            bMagnitude += dequantizedVector.get(i) * dequantizedVector.get(i);
        }

        return new float[]{sum, bMagnitude};
    }

    static void nvqShuffleQueryInPlace4bit(ArrayVectorFloat vector) {
        // To understand this shuffle, see nvqDequantize4bit
        var shuffle = VectorShuffle.fromValues(FloatVector.SPECIES_512,
                0, 2, 4, 6, 8, 10, 12, 14,
                1, 3, 5, 7, 9, 11, 13, 15);
        int vectorizedLength = FloatVector.SPECIES_512.loopBound(vector.length());
        var arr = vector.get();
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_512.length()) {
            FloatVector.fromArray(FloatVector.SPECIES_512, arr, i).rearrange(shuffle).intoArray(arr, i);;
        }
        // There's no need to shuffle the tail
    }

    //---------------------------------------------
    // NVQ quantization instructions end here
    //---------------------------------------------

}
