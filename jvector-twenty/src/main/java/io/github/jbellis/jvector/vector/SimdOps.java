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
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteOrder;
import java.util.List;

final class SimdOps {
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

    static void divInPlace(ArrayVectorFloat vector, float divisor) {
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length());

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector.get(), i);
            var divResult = a.div(divisor);
            divResult.intoArray(vector.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length(); i++) {
            vector.set(i, vector.get(i) / divisor);
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
            sum = sum.add(a.mul(b));
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
            sum = sum.add(a.mul(b));
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
            sum = sum.add(a.mul(b));
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
            sum = sum.add(a.mul(b));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i)
            res += v1.get(v1offset + i) * v2.get(v2offset + i);

        return res;
    }

    static int dotProduct(ArrayVectorByte v1, ArrayVectorByte v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        var sum = IntVector.zero(IntVector.SPECIES_256);
        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length());

        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1.get(), i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2.get(), i).castShape(IntVector.SPECIES_256, 0);
            sum = sum.add(a.mul(b));
        }

        int res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            res += v1.get(i) * v2.get(i);
        }

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
            vsum = vsum.add(a.mul(b));
            vaMagnitude = vaMagnitude.add(a.mul(a));
            vbMagnitude = vbMagnitude.add(b.mul(b));
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

    static float cosineSimilarity(ArrayVectorByte v1, ArrayVectorByte v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = IntVector.zero(IntVector.SPECIES_256);
        var vaMagnitude = IntVector.zero(IntVector.SPECIES_256);
        var vbMagnitude = IntVector.zero(IntVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length());
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1.get(), i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2.get(), i).castShape(IntVector.SPECIES_256, 0);
            vsum = vsum.add(a.mul(b));
            vaMagnitude = vaMagnitude.add(a.mul(a));
            vbMagnitude = vbMagnitude.add(b.mul(b));
        }

        int sum = vsum.reduceLanes(VectorOperators.ADD);
        int aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        int bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            sum += v1.get(i) * v2.get(i);
            aMagnitude += v1.get(i) * v1.get(i);
            bMagnitude += v2.get(i) * v2.get(i);
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
            sum = sum.add(diff.mul(diff));
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
            sum = sum.add(diff.mul(diff));
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
            sum = sum.add(diff.mul(diff));
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
            sum = sum.add(diff.mul(diff));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; i < length; ++i) {
            var diff = v1.get(v1offset + i) - v2.get(v2offset + i);
            res += diff * diff;
        }

        return res;
    }

    static int squareDistance(ArrayVectorByte v1, ArrayVectorByte v2) {
        if (v1.length() != v2.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vdiffSumSquared = IntVector.zero(IntVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length());
        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1.get(), i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2.get(), i).castShape(IntVector.SPECIES_256, 0);

            var diff = a.sub(b);
            vdiffSumSquared = vdiffSumSquared.add(diff.mul(diff));
        }

        int diffSumSquared = vdiffSumSquared.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length(); i++) {
            diffSumSquared += (v1.get(i) - v2.get(i)) * (v1.get(i) - v2.get(i));
        }

        return diffSumSquared;
    }

    static void addInPlace64(ArrayVectorFloat v1, ArrayVectorFloat v2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1.get(), 0);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2.get(), 0);
        a.add(b).intoArray(v1.get(), 0);
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

    static VectorFloat<?> sub(ArrayVectorFloat lhs, ArrayVectorFloat rhs) {
        if (lhs.length() != rhs.length()) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        ArrayVectorFloat result = new ArrayVectorFloat(lhs.length());
        int vectorizedLength = (lhs.length() / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, lhs.get(), i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, rhs.get(), i);
            var subResult = a.sub(b);
            subResult.intoArray(result.get(), i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < lhs.length(); i++) {
            result.set(i, lhs.get(i) - rhs.get(i));
        }

        return result;
    }
}
