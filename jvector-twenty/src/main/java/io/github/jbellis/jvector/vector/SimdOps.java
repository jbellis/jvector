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

import io.github.jbellis.jvector.util.MathUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import jdk.incubator.vector.*;

import java.util.List;

final class SimdOps {

    static final boolean HAS_AVX512 = IntVector.SPECIES_PREFERRED == IntVector.SPECIES_512;
    static final IntVector BYTE_TO_INT_MASK_512 = IntVector.broadcast(IntVector.SPECIES_512, 0xff);
    static final IntVector BYTE_TO_INT_MASK_256 = IntVector.broadcast(IntVector.SPECIES_256, 0xff);

    static final ThreadLocal<int[]> scratchInt512 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_512.length()]);
    static final ThreadLocal<int[]> scratchInt256 = ThreadLocal.withInitial(() -> new int[IntVector.SPECIES_256.length()]);

    static VectorSpecies<Byte> byteSpecies;
    // Dynamically set species for ByteVector based on the platform's SIMD width
    static {
        if (FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_512) {
            byteSpecies = ByteVector.SPECIES_128;  // 512-bit SIMD width: 16 lanes, each a byte
        } else if (FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_256) {
            byteSpecies = ByteVector.SPECIES_64;   // 256-bit SIMD width: 8 lanes
        } else {
            throw new IllegalStateException("Unsupported SIMD width for ByteVector species.");
        }
    }

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

    /*
     Vectorized fast exponential
     https://codingforspeed.com/using-faster-exponential-approximation/
     */
    public static FloatVector fastExp(FloatVector x) {
        x = x.div(1024).add(1.f);
        x = x.mul(x); x = x.mul(x); x = x.mul(x); x = x.mul(x);
        x = x.mul(x); x = x.mul(x); x = x.mul(x); x = x.mul(x);
        x = x.mul(x); x = x.mul(x);
        return x;
    }

    /*
     Vectorized fast natural logarithm on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5.
     https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
     */
    public static FloatVector fastLog(FloatVector x, VectorSpecies<Float> fSpecies) {
        IntVector temp = x.reinterpretAsInts();
        var e = temp.sub(0x3f2aaaab).and(0xff800000);
        var m = temp.sub(e).reinterpretAsFloats();
        var i = e.castShape(fSpecies, 0).reinterpretAsFloats().mul(1.19209290e-7f);  // 0x1.0p-23

        /* m in [2/3, 4/3] */
        var f = m.sub(1.f);
        var s = f.mul(f);

        /* Compute log1p(f) for f in [-1/3, 1/3] */
        var r = f.fma(0.230836749f, -0.279208571f);  // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        var t = f.fma(0.331826031f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2)
        r = r.fma(s, t);
        r = r.fma(s, f);
        r = i.fma(FloatVector.broadcast(fSpecies, 0.693147182f), r); // 0x1.62e430p-1 // log(2)
        return r;
    }

    static FloatVector forwardKumaraswamy(FloatVector vector, float a, float b, VectorSpecies<Float> fSpecies) {
        var temp = fastExp(fastLog(vector, fSpecies).mul(a)).neg().add(1.f);   // 1 - v ** a
        return fastExp(fastLog(temp, fSpecies).mul(b)).neg().add(1.f);        // 1 - v ** b
    }

    static float forwardKumaraswamy(float value, float a, float b) {
        var temp = 1.f - MathUtil.fastExp(MathUtil.fastLog(value) * a);   // 1 - v ** a
        return 1.f - MathUtil.fastExp(MathUtil.fastLog(temp) * b);        // 1 - v ** b
    }

//    static FloatVector inverseKumaraswamy(FloatVector vector, float a, float b, VectorSpecies<Float> fSpecies) {
//        var temp = fastExp(fastLog(vector.neg().add(1.f), fSpecies).div(b));  // (1 - v) ** (1 / b)
//        return fastExp(fastLog(temp.neg().add(1.f), fSpecies).div(a));        // (1 - v) ** (1 / a)
//    }

    static FloatVector inverseKumaraswamy(FloatVector vector, float a, float b, VectorSpecies<Float> fSpecies) {
        float invA = 1.0f / a; // precompute
        float invB = 1.0f / b;

        // Store intermediates in registers
        var oneMinusV = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.0f).sub(vector);

        // Use broadcasting
        FloatVector exponentB = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, invB);
        oneMinusV = oneMinusV.pow(exponentB);  // In-place operation (no heap allocation)

        var oneMinusVToBSub = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.0f).sub(oneMinusV);

        // Compute (1 - (1 - v) ** (1/b)) ** (1/a) and store to register
        FloatVector exponentA = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, invA);
        return oneMinusVToBSub.pow(exponentA);
    }

    static float inverseKumaraswamy(float value, float a, float b) {
        var temp = MathUtil.fastExp(MathUtil.fastLog(1.f - value) / b);   // (1 - v) ** (1 / b)
        return MathUtil.fastExp(MathUtil.fastLog(1.f - temp) / a);        // (1 - v) ** (1 / a)
    }

    static FloatVector nvqDequantizeUnnormalized4bitPart1(ByteVector bytes, float a, float b) {
        /*
         * bytes:      |  0   |  1   |  2   |  3   |  4   |  5     |  6     |  7     |
         * half-bytes: | 0  1 | 2  3 | 4  5 | 6  7 | 8  9 | 10  11 | 12  13 | 14  15 |
         *
         * Part1: 1st pass of original array (this is returned)
         * & 0xf:      | 0    | 2    | 4    | 6    | 8    | 10     | 12     | 14     |
         * Part2: 2nd pass of original array
         * lS 4:       | 1    | 3    | 5    | 7    | 9    | 11     | 13     | 15     |
         */
        var arr = bytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.AND, 0xf)
                .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0)
                .reinterpretAsFloats()
                .div(15.f);
        return inverseKumaraswamy(arr, a, b, FloatVector.SPECIES_256);
    }

    static FloatVector nvqDequantizeUnnormalized4bitPart2(ByteVector bytes, float a, float b) {
        /*
         * bytes:      |  0   |  1   |  2   |  3   |  4   |  5     |  6     |  7     |
         * half-bytes: | 0  1 | 2  3 | 4  5 | 6  7 | 8  9 | 10  11 | 12  13 | 14  15 |
         *
         * Part1: 1st pass of original array
         * & 0xf:      | 0    | 2    | 4    | 6    | 8    | 10     | 12     | 14     |
         * Part2: 2nd pass of original array (this is returned)
         * lS 4:       | 1    | 3    | 5    | 7    | 9    | 11     | 13     | 15     |
         */
        var arr = bytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.AND, 0xff)
                .lanewise(VectorOperators.LSHR, 4)
                .convertShape(VectorOperators.I2F, FloatVector.SPECIES_256, 0)
                .reinterpretAsFloats()
                .div(15.f);
        return inverseKumaraswamy(arr, a, b, FloatVector.SPECIES_256);
    }

    static float nvqDequantizeUnnormalized4bitTailPart1(Byte byteValue, float a, float b) {
        var intValue = Byte.toUnsignedInt(byteValue);
        float value = intValue & 0xf;
        return inverseKumaraswamy(value / 15.f, a, b);
    }

    static float nvqDequantizeUnnormalized4bitTailPart2(Byte byteValue, float a, float b) {
        var intValue = Byte.toUnsignedInt(byteValue);
        float value = intValue << 4;
        return inverseKumaraswamy(value / 15.f, a, b);
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
            var subResult = nvqDequantizeUnnormalized4bitPart1(arr, a, b);
            subResult.intoArray(resArr, 2 * i);

            // 2nd pass
            subResult = nvqDequantizeUnnormalized4bitPart2(arr, a, b);
            subResult.intoArray(resArr, 2 * i + ByteVector.SPECIES_64.length());
        }

        // Process the tail
        if (vectorizedLength < bytes.length()) {
            for (int i = vectorizedLength; i < bytes.length(); i++) {
                resArr[2 * i] = nvqDequantizeUnnormalized4bitTailPart1(bytes.get(i), a, b);
                if (2 * i + 1 < resArr.length) {
                    resArr[2 * i + 1] = nvqDequantizeUnnormalized4bitTailPart2(bytes.get(i), a, b);
                }
            }
        }
    }

    static FloatVector nvqDequantizeUnnormalized8bit(ByteVector bytes, float a, float b) {
        var arr = bytes.convertShape(VectorOperators.B2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.AND, 0xff)
                .convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0)
                .reinterpretAsFloats()
                .lanewise(VectorOperators.DIV, 255.0f);

        return inverseKumaraswamy(arr, a, b, FloatVector.SPECIES_PREFERRED);
    }

    static void nvqDequantizeUnnormalized8bit(ArrayByteSequence bytes, float a, float b, ArrayVectorFloat res) {
        var resArr = res.get();

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(bytes.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var arr = nvqDequantizeUnnormalized8bit(ByteVector.fromArray(ByteVector.SPECIES_64, bytes.get(), i), a, b);
            arr.intoArray(resArr, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < bytes.length(); i++) {
            float value = bytes.get(i);
            resArr[i] = inverseKumaraswamy(value / 255.f, a, b);
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
        var resArr = destination.get();

        int vectorizedLength = ByteVector.SPECIES_128.loopBound(bytes.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_128.length()) {
            var arr = nvqDequantizeUnnormalized8bit(ByteVector.fromArray(ByteVector.SPECIES_128, bytes.get(), i), a, b);
            arr = arr.mul(scale).add(bias);
            arr.intoArray(resArr, i);
        }

        // Process the tail
        float value;
        for (int i = vectorizedLength; i < bytes.length(); i++) {
            value = bytes.get(i);
            resArr[i] = scale * inverseKumaraswamy(value / 255.f, a, b) + bias;
        }
    }

    static void nvqDequantize4bit(ArrayByteSequence bytes, float a, float b, float scale, float bias, ArrayVectorFloat destination) {
        /*
         * bytes:      |  0   |  1   |  2   |  3   |  4   |  5     |  6     |  7     |
         * half-bytes: | 0  1 | 2  3 | 4  5 | 6  7 | 8  9 | 10  11 | 12  13 | 14  15 |
         *
         * 1st pass of original array:
         * & 0xf:      | 0    | 2    | 4    | 6    | 8    | 10     | 12     | 14     |
         * 2nd pass of original array:
         * lS 4:       | 1    | 3    | 5    | 7    | 9    | 11     | 13     | 15     |
         */
        var resArr = destination.get();

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(bytes.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var arr = ByteVector.fromArray(ByteVector.SPECIES_64, bytes.get(), i);
            // 1st pass
            var subResult = nvqDequantizeUnnormalized4bitPart1(arr, a, b);
            subResult = subResult.mul(scale).add(bias);
            subResult.intoArray(resArr, 2 * i);

            // 2nd pass
            subResult = nvqDequantizeUnnormalized4bitPart2(arr, a, b);
            subResult = subResult.mul(scale).add(bias);
            subResult.intoArray(resArr, 2 * i + ByteVector.SPECIES_64.length());
        }

        // Process the tail
        if (vectorizedLength < bytes.length()) {
            for (int i = vectorizedLength; i < bytes.length(); i++) {
                resArr[2 * i] = scale * nvqDequantizeUnnormalized4bitTailPart1(bytes.get(i), a, b) + bias;
                if (2 * i + 1 < resArr.length) {
                    resArr[2 * i + 1] = scale * nvqDequantizeUnnormalized4bitTailPart2(bytes.get(i), a, b) + bias;
                }
            }
        }
    }

    static void nvqQuantizeNormalized8bit(ArrayVectorFloat vector, float a, float b, ArrayByteSequence destination) {
        final int constant = (1 << 8) - 1;
        final int vectorizedLength = FloatVector.SPECIES_512.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_128.indexInRange(0, FloatVector.SPECIES_512.length());

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_512.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_512, vector.get(), i);
            arr = forwardKumaraswamy(arr, a, b, FloatVector.SPECIES_512);
            var bytes = arr.mul(constant).add(0.5f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0)
                    .reinterpretAsBytes();
            bytes.intoArray(destination.get(), i, mask);
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d++) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = forwardKumaraswamy(value, a, b);
            int quantizedValue = Math.min(Math.max(0, Math.round(constant * value)), constant);
            destination.set(d, (byte) quantizedValue);
        }
    }

    static void nvqQuantizeNormalized4bit(ArrayVectorFloat vector, float a, float b, ArrayByteSequence destination) {
        final int constant = (1 << 4) - 1;

        final var shuffle = VectorShuffle.fromValues(FloatVector.SPECIES_256, 1, 0, 3, 2, 5, 4, 7, 6);
        final var finalShuffle = VectorShuffle.fromValues(ByteVector.SPECIES_64, 0, 2, 4, 6, 1, 3, 5, 7);

        final int vectorizedLength = FloatVector.SPECIES_256.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_64.indexInRange(0, FloatVector.SPECIES_256.length() / 2);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_256.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), i);
            arr = forwardKumaraswamy(arr, a, b, FloatVector.SPECIES_256);
            arr = arr.mul(constant).add(0.5f);

            var bytesEven = arr.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var arrOdd = arr.rearrange(shuffle);
            var bytesOdd = arrOdd.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            bytesOdd = bytesOdd.lanewise(VectorOperators.LSHL, 4);

            bytesEven.add(bytesOdd).rearrange(finalShuffle).intoArray(destination.get(), i / 2, mask);
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d += 2) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = forwardKumaraswamy(value, a, b);
            int quantizedValue0 = Math.min(Math.max(0, Math.round(constant * value)), constant);
            int quantizedValue1;
            if (d + 1 < vector.length()) {
                value = vector.get(d + 1);
                value = forwardKumaraswamy(value, a, b);
                quantizedValue1 = Math.min(Math.max(0, Math.round(constant * value)), constant);
            } else {
                quantizedValue1 = 0;
            }
            destination.set(d / 2, (byte) ((quantizedValue1 << 4) + quantizedValue0));
        }
    }

    static float nvqLoss(ArrayVectorFloat vector, float a, float b, int nBits) {
        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        float squaredSum = 0.f;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var recArr = forwardKumaraswamy(arr, a, b, FloatVector.SPECIES_PREFERRED);
            recArr = recArr.mul(constant).add(0.5f)
                    .convertShape(VectorOperators.F2I, IntVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsInts()
                    .convertShape(VectorOperators.I2F, FloatVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsFloats()
                    .div(constant);
            recArr = inverseKumaraswamy(recArr, a, b, FloatVector.SPECIES_PREFERRED);

            var diff = arr.sub(recArr);
            squaredSum += diff.mul(diff).reduceLanes(VectorOperators.ADD);
        }

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = forwardKumaraswamy(value, a, b);
            recValue = Math.min(Math.max(0, Math.round(constant * recValue)), constant);
            recValue /= constant;
            recValue = inverseKumaraswamy(recValue, a, b);

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    /**
     * Compute the squared L2 distance for 8-bit NVQ
     * Each sub-vector of query vector (full resolution) will be compared to NVQ quantized sub-vectors that were
     * first de-meaned by subtracting the global mean.
     *
     * The squared L2 distance is calculated between the query and quantized sub-vectors as follows:
     *
     * |query - vector|^2 \approx |query - (scale * quantized + bias + globalMean)|^2
     *                          = |(query - globalMean) - scale * quantized + bias|^2
     *
     * @param vector The shifted query (precomputed query - globalMean)
     * @param quantizedVector A quantized vector from the index
     * @return The square L2 distance
     */
    static float nvqSquareDistance8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias) {
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_512);

        int vectorizedLength = ByteVector.SPECIES_128.loopBound(quantizedVector.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_128.length()) {
            var v1 = FloatVector.fromArray(FloatVector.SPECIES_512, vector.get(), i);

            // dequantize
            var v2 = nvqDequantizeUnnormalized8bit(ByteVector.fromArray(ByteVector.SPECIES_128, quantizedVector.get(), i), a, b);
            v2 = v2.mul(scale).add(bias);

            var diff = v1.sub(v2);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2, diff;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = quantizedVector.get(i);
            value2 = scale * inverseKumaraswamy(value2 / 255.f, a, b) + bias;
            diff = vector.get(i) - value2;
            squaredSum += MathUtil.square(diff);
        }

        return squaredSum;
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
    static float nvqSquareDistance4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias) {
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(quantizedVector.length());
        FloatVector v1, v2, diff;
        ByteVector bv2;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            bv2 = ByteVector.fromArray(ByteVector.SPECIES_64, quantizedVector.get(), i);

            // 1st pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i);

            v2 = nvqDequantizeUnnormalized4bitPart1(bv2, a, b);
            v2 = v2.mul(scale).add(bias);

            diff = v1.sub(v2);
            squaredSumVec = diff.fma(diff, squaredSumVec);

            // 2nd pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i + ByteVector.SPECIES_64.length());

            v2 = nvqDequantizeUnnormalized4bitPart2(bv2, a, b);
            v2 = v2.mul(scale).add(bias);

            diff = v1.sub(v2);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        if (vectorizedLength < quantizedVector.length()) {
            float value2;
            for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
                value2 = scale * nvqDequantizeUnnormalized4bitTailPart1(quantizedVector.get(i), a, b) + bias;
                squaredSum += MathUtil.square(vector.get(2 * i) - value2);
                if (2 * i + 1 < quantizedVector.length()) {
                    value2 = scale * nvqDequantizeUnnormalized4bitTailPart2(quantizedVector.get(i), a, b) + bias;
                    squaredSum += MathUtil.square(vector.get(2 * i + 1) - value2);
                }
            }
        }

        return squaredSum;
    }

    static float nvqDotProduct8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias, float vectorSum) {
        FloatVector dotProdVec = FloatVector.zero(FloatVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(quantizedVector.length());

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), i);

            // dequantize
            var v2 = nvqDequantizeUnnormalized8bit(ByteVector.fromArray(ByteVector.SPECIES_64, quantizedVector.get(), i), a, b);

            dotProdVec = v1.fma(v2, dotProdVec);
        }

        float dotProd = dotProdVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = quantizedVector.get(i);
            value2 = inverseKumaraswamy(value2 / 255.f, a, b);
            dotProd += vector.get(i) - value2;
        }

        return scale * dotProd + bias * vectorSum;
    }

    static float nvqDotProduct4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias, float vectorSum) {
        FloatVector dotProdVec = FloatVector.zero(FloatVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(quantizedVector.length());
        FloatVector v1, v2;
        ByteVector bv2;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            bv2 = ByteVector.fromArray(ByteVector.SPECIES_64, quantizedVector.get(), i);

            // 1st pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i);
            v2 = nvqDequantizeUnnormalized4bitPart1(bv2, a, b);

            dotProdVec = v1.fma(v2, dotProdVec);

            // 2nd pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i + ByteVector.SPECIES_64.length());
            v2 = nvqDequantizeUnnormalized4bitPart2(bv2, a, b);

            dotProdVec = v1.fma(v2, dotProdVec);
        }

        float dotProd = dotProdVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        if (vectorizedLength < quantizedVector.length()) {
            float value2;
            for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
                value2 = nvqDequantizeUnnormalized4bitTailPart1(quantizedVector.get(i), a, b);
                dotProd += vector.get(2 * i) * value2;
                if (2 * i + 1 < quantizedVector.length()) {
                    value2 = nvqDequantizeUnnormalized4bitTailPart2(quantizedVector.get(i), a, b);
                    dotProd += vector.get(2 * i + 1) * value2;
                }
            }
        }

        return scale * dotProd + bias * vectorSum;
    }

    static float[] nvqCosine8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias, ArrayVectorFloat centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);

            // dequantize
            var vb = nvqDequantizeUnnormalized8bit(ByteVector.fromArray(byteSpecies, quantizedVector.get(), i), a, b);
            vb = vb.mul(scale).add(bias);

            var vCentroid = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, centroid.get(), i);
            vb = vb.add(vCentroid);

            vsum = va.fma(vb, vsum);
            vbMagnitude = vb.fma(vb, vbMagnitude);
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value2 = quantizedVector.get(i);
            value2 = scale * inverseKumaraswamy(value2 / 255.f, a, b) + bias + centroid.get(i);
            sum += vector.get(i) * value2;
            bMagnitude += value2 * value2;
        }

        return new float[]{sum, bMagnitude};
    }

    static float[] nvqCosine4bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector, float a, float b, float scale, float bias, ArrayVectorFloat centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vDotProduct = FloatVector.zero(FloatVector.SPECIES_256);
        var vbNorm = FloatVector.zero(FloatVector.SPECIES_256);


        int vectorizedLength = ByteVector.SPECIES_64.loopBound(quantizedVector.length());
        FloatVector v1, v2, vCentroid;
        ByteVector bv2;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            bv2 = ByteVector.fromArray(ByteVector.SPECIES_64, quantizedVector.get(), i);

            // 1st pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i);
            vCentroid = FloatVector.fromArray(FloatVector.SPECIES_256, centroid.get(), 2 * i);

            v2 = nvqDequantizeUnnormalized4bitPart1(bv2, a, b);
            v2 = v2.mul(scale).add(bias);
            v2 = v2.add(vCentroid);

            vDotProduct = v1.fma(v2, vDotProduct);
            vbNorm = v2.fma(v2, vbNorm);

            // 2nd pass
            v1 = FloatVector.fromArray(FloatVector.SPECIES_256, vector.get(), 2 * i + ByteVector.SPECIES_64.length());
            vCentroid = FloatVector.fromArray(FloatVector.SPECIES_256, centroid.get(), 2 * i + ByteVector.SPECIES_64.length());

            v2 = nvqDequantizeUnnormalized4bitPart2(bv2, a, b);
            v2 = v2.mul(scale).add(bias);
            v2 = v2.add(vCentroid);

            vDotProduct = v1.fma(v2, vDotProduct);
            vbNorm = v2.fma(v2, vbNorm);
        }

        float dotProduct = vDotProduct.reduceLanes(VectorOperators.ADD);
        float quantizedSquaredNorm = vbNorm.reduceLanes(VectorOperators.ADD);

        // Process the tail
        if (vectorizedLength < quantizedVector.length()) {
            float value2;
            for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
                value2 = scale * nvqDequantizeUnnormalized4bitTailPart1(quantizedVector.get(i), a, b) + bias;
                value2 += centroid.get(2 * i);
                dotProduct += vector.get(2 * i) * value2;
                quantizedSquaredNorm += value2 * value2;
                if (2 * i + 1 < quantizedVector.length()) {
                    value2 = scale * nvqDequantizeUnnormalized4bitTailPart2(quantizedVector.get(i), a, b) + bias;
                    value2 += centroid.get(2 * i + 1);
                    dotProduct += vector.get(2 * i + 1) * value2;
                    quantizedSquaredNorm += value2 * value2;
                }
            }
        }
        return new float[]{dotProduct, quantizedSquaredNorm};
    }

    static void nvqShuffleQueryInPlace4bit(ArrayVectorFloat vector) {
        // To understand this shuffle, see nvqDequantize4bit
        final var shuffle = VectorShuffle.fromValues(FloatVector.SPECIES_512,
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
    // NVQ instructions end here
    //---------------------------------------------
}
