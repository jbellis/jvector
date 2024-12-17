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

    static FloatVector const0693147182f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.693147182f);
    static FloatVector const1f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f);
    static FloatVector const05f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.5f);

    /*
     Fast exponential based on:
     https://stackoverflow.com/questions/47025373/fastest-implementation-of-the-natural-exponential-function-using-sse
     The Remez polynomial has been modified for higher accuracy
     */
    public static FloatVector fastExp(FloatVector x){
        // approximation of exp(x)
        // A0 + x * (A1 + x * (A2 + x * (a3 + x * (a4 + x * (a5 + x * (a6 + x * a7))))));
        final float invlog2e = 1.442695041f;  // 1 / log2(e)
        final float expCvt = 12582912.0f;  // 1.5 * (1 << 23)
        final float expA0 = 0.9999993887682104f;
        final float expA1 = 0.6931186232012877f;
        final float expA2 = 0.2402301551437674f;
        final float expA3 = 0.05593479631997887f;
        final float expA4 = 0.009651907610706037f;

        /* exp(x) = 2^i * 2^f; i = rint (log2(e) * x), -0.5 <= f <= 0.5 */
        var t = x.mul(FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, invlog2e));  // t = x / log2(e)
        var r = t.add(expCvt).sub(expCvt);  // r = round(t)
        var f = t.sub(r);  // f = t - round(t)
        var i = r.castShape(IntVector.SPECIES_PREFERRED, 0).reinterpretAsInts(); // i = (int) r

        var temp = f.fma(FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, expA4), FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, expA3));
        temp = temp.fma(f, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, expA2));
        temp = temp.fma(f, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, expA1));
        temp = temp.fma(f, FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, expA0));

        i = i.lanewise(VectorOperators.MAX, -126);
        var j = i.lanewise(VectorOperators.LSHL, 23);
        temp = temp.reinterpretAsInts().add(j).reinterpretAsFloats();  // temp = temp * 2^i
        return temp;
    }

    /*
     Vectorized fast natural logarithm on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5.
     https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
     */
    public static FloatVector fastLog(FloatVector x) {
        IntVector temp = x.reinterpretAsInts();
        var e = temp.sub(0x3f2aaaab).and(0xff800000);
        FloatVector m = temp.sub(e).reinterpretAsFloats();
        var i = e.castShape(FloatVector.SPECIES_PREFERRED, 0).reinterpretAsFloats();
        i = i.mul(1.19209290e-7f);  // 0x1.0p-23

        /* m in [2/3, 4/3] */
        var f = m.sub(1.f);
        var s = f.mul(f);

        /* Compute log1p(f) for f in [-1/3, 1/3] */
        var r = f.fma(0.230836749f, -0.279208571f);  // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        var t = f.fma(0.331826031f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2)
        r = r.fma(s, t);
        r = r.fma(s, f);
        return i.fma(const0693147182f, r); // 0x1.62e430p-1 // log(2)
    }

    static FloatVector logistic(FloatVector vector, float alpha, float x0) {
       var temp = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -alpha).mul(vector.sub(x0));
       temp = fastExp(temp);
       temp = const1f.add(temp);
       return const1f.div(temp);
    }

    static float logistic(float value, float alpha, float x0) {
        var temp = -alpha * (value - x0);
        temp = MathUtil.fastExp(temp);
        return 1 / (1 + temp);
    }

    static FloatVector logit(FloatVector vector, float inverseAlpha, float x0) {
        var temp = vector.div(const1f.sub(vector));
        return fastLog(temp).fma(inverseAlpha, x0);
    }

    static float logit(float value, float alpha, float x0) {
        var temp = value / (1 - value);
        return MathUtil.fastLog(temp) / alpha + x0;
    }

    static FloatVector nvqDequantize8bit(ByteVector bytes, float inverseAlpha, float x0, float logisticScale, float logisticBias, int part) {
        var arr = bytes.reinterpretAsInts()
                .lanewise(VectorOperators.LSHR, 8 * part)
                .lanewise(VectorOperators.AND, 0xff)
                .convert(VectorOperators.I2F, 0)
                .reinterpretAsFloats();

        arr = arr.fma(logisticScale, logisticBias);
        return logit(arr, inverseAlpha, x0);
    }

    static void nvqDequantize8bit(ArrayByteSequence bytes, float alpha, float x0, float logisticScale, float logisticBias, ArrayVectorFloat destination) {
        var resArr = destination.get();

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(bytes.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, bytes.get(), i);

            for (int j = 0; j < 4; j++) {
                var arr = nvqDequantize8bit(byteArr, alpha, x0, logisticScale, logisticBias, j);
                arr.intoArray(resArr, i + floatStep * j);
            }
        }

        // Process the tail
        float value;
        for (int i = vectorizedLength; i < bytes.length(); i++) {
            value = bytes.get(i);
            resArr[i] = logit(value / 255.f, alpha, x0);
        }
    }

    static void nvqQuantize8bit(ArrayVectorFloat vector, float alpha, float x0, float minValue, float maxValue, ArrayByteSequence destination) {
        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final var mask = ByteVector.SPECIES_PREFERRED.indexInRange(0, FloatVector.SPECIES_PREFERRED.length());

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var scaledX0 = x0 * delta;
        var logisticBias = logistic(minValue, scaledAlpha, scaledX0);
        var invLogisticScale = 255 / (logistic(maxValue, scaledAlpha, scaledX0) - logisticBias);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            arr = logistic(arr, scaledAlpha, scaledX0);
            arr = arr.sub(logisticBias).mul(invLogisticScale);
            var bytes = arr.add(const05f)
                    .convertShape(VectorOperators.F2B, ByteVector.SPECIES_PREFERRED, 0)
                    .reinterpretAsBytes();
            bytes.intoArray(destination.get(), i, mask);
        }

        // Process the tail
        for (int d = vectorizedLength; d < vector.length(); d++) {
            // Ensure the quantized value is within the 0 to constant range
            float value = vector.get(d);
            value = logistic(value, scaledAlpha, scaledX0);
            value = (value - logisticBias) * invLogisticScale;
            int quantizedValue = Math.round(value);
            destination.set(d, (byte) quantizedValue);
        }
    }

    static float nvqLoss(ArrayVectorFloat vector, float alpha, float x0, float minValue, float maxValue, int nBits) {
        int constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logistic(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logistic(maxValue, scaledAlpha, scaledX0) - logisticBias) / constant;
        var invLogisticScale = 1 / logisticScale;

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var recArr = logistic(arr, scaledAlpha, scaledX0);
            recArr = recArr.sub(logisticBias).mul(invLogisticScale);
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma(logisticScale, logisticBias);
            recArr = logit(recArr, invScaledAlpha, scaledX0);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = logistic(value, scaledAlpha, scaledX0);
            recValue = (recValue - logisticBias) * invLogisticScale;
            recValue = Math.round(recValue);
            recValue = logisticScale * recValue + logisticBias;
            recValue = logit(recValue, scaledAlpha, scaledX0);

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    static float nvqUniformLoss(ArrayVectorFloat vector, float minValue, float maxValue, int nBits) {
        float constant = (1 << nBits) - 1;
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var arr = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var recArr = arr.sub(minValue).mul(constant / (maxValue - minValue));
            recArr = recArr.add(const05f)
                    .convert(VectorOperators.F2I, 0)
                    .reinterpretAsInts()
                    .convert(VectorOperators.I2F, 0)
                    .reinterpretAsFloats();
            recArr = recArr.fma((maxValue - minValue) / constant, minValue);

            var diff = arr.sub(recArr);
            squaredSumVec = diff.fma(diff, squaredSumVec);
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value, recValue;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value = vector.get(i);

            recValue = (value - minValue) / (maxValue - minValue);
            recValue = Math.round(constant * recValue) / constant;
            recValue = recValue / (maxValue - minValue) + minValue;

            squaredSum += MathUtil.square(value - recValue);
        }

        return squaredSum;
    }

    static float nvqSquareDistance8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector,
                                       float alpha, float x0, float minValue, float maxValue) {
        FloatVector squaredSumVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logistic(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logistic(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var v1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var diff = v1.sub(v2);
                squaredSumVec = diff.fma(diff, squaredSumVec);
            }
        }

        float squaredSum = squaredSumVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2, diff;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = logisticScale * value2 + logisticBias;
            value2 = logit(value2, scaledAlpha, scaledX0);
            diff = vector.get(i) - value2;
            squaredSum += MathUtil.square(diff);
        }

        return squaredSum;
    }

    static float nvqDotProduct8bit(ArrayVectorFloat vector, ArrayByteSequence quantizedVector,
                                   float alpha, float x0, float minValue, float maxValue) {
        FloatVector dotProdVec = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(quantizedVector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logistic(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logistic(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;


        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var v1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i + floatStep * j);
                var v2 = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);
                dotProdVec = v1.fma(v2, dotProdVec);
            }
        }

        float dotProd = dotProdVec.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < quantizedVector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = logisticScale * value2 + logisticBias;
            value2 = logit(value2, scaledAlpha, scaledX0);
            dotProd += vector.get(i) * value2;
        }

        return dotProd;
    }

    static float[] nvqCosine8bit(ArrayVectorFloat vector,
                                 ArrayByteSequence quantizedVector, float alpha, float x0, float minValue, float maxValue,
                                 ArrayVectorFloat centroid) {
        if (vector.length() != centroid.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var delta = maxValue - minValue;
        var scaledAlpha = alpha / delta;
        var invScaledAlpha = 1 / scaledAlpha;
        var scaledX0 = x0 * delta;
        var logisticBias = logistic(minValue, scaledAlpha, scaledX0);
        var logisticScale = (logistic(maxValue, scaledAlpha, scaledX0) - logisticBias) / 255;

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = ByteVector.SPECIES_PREFERRED.loopBound(vector.length());
        int floatStep = FloatVector.SPECIES_PREFERRED.length();

        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_PREFERRED.length()) {
            var byteArr = ByteVector.fromArray(ByteVector.SPECIES_PREFERRED, quantizedVector.get(), i);

            for (int j = 0; j < 4; j++) {
                var va = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i + floatStep * j);
                var vb = nvqDequantize8bit(byteArr, invScaledAlpha, scaledX0, logisticScale, logisticBias, j);

                var vCentroid = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, centroid.get(), i + floatStep * j);
                vb = vb.add(vCentroid);

                vsum = va.fma(vb, vsum);
                vbMagnitude = vb.fma(vb, vbMagnitude);
            }
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        float value2;
        for (int i = vectorizedLength; i < vector.length(); i++) {
            value2 = Byte.toUnsignedInt(quantizedVector.get(i));
            value2 = logisticScale * value2 + logisticBias;
            value2 = logit(value2, scaledAlpha, scaledX0) + centroid.get(i);
            sum += vector.get(i) * value2;
            bMagnitude += value2 * value2;
        }

        // TODO can we avoid returning a new array?
        return new float[]{sum, bMagnitude};
    }

    static void transpose(float[] arr, int first, int last, int nRows) {
        final int mn1 = (last - first - 1);
        final int n   = (last - first) / nRows;
        boolean[] visited = new boolean[last - first];
        float temp;
        int cycle = first;
        while (++cycle != last) {
            if (visited[cycle - first])
                continue;
            int a = cycle - first;
            do  {
                a = a == mn1 ? mn1 : (n * a) % mn1;
                temp = arr[first + a];
                arr[first + a] = arr[cycle];
                arr[cycle] = temp;
                visited[a] = true;
            } while ((first + a) != cycle);
        }
    }

    static void nvqShuffleQueryInPlace8bit(ArrayVectorFloat vector) {
        // To understand this shuffle, see nvqDequantize8bit
        var arr = vector.get();

        final int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());
        final int step = FloatVector.SPECIES_PREFERRED.length() * 4;

        for (int i = 0; i + step <= vectorizedLength; i += step) {
            transpose(arr, i, i + step, 4);
        }
    }

    //---------------------------------------------
    // NVQ instructions end here
    //---------------------------------------------
}
