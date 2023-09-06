package com.github.jbellis.jvector.vector;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SimdOps {

    private static final Logger logger = LoggerFactory.getLogger(SimdOps.class);

    public static float sum(float[] vector) {
        var sum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(vector.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector, i);
            sum = sum.add(a);
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < vector.length; i++) {
            res += vector[i];
        }

        return res;
    }

    public static float[] sum(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length;
        float[] sum = new float[dimension];

        // Process each vector from the list
        for (float[] vector : vectors) {
            addInPlace(sum, vector);
        }

        return sum;
    }

    public static void divInPlace(float[] vector, float divisor) {
        int vectorizedLength = (vector.length / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector, i);
            var divResult = a.div(divisor);
            divResult.intoArray(vector, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length; i++) {
            vector[i] = vector[i] / divisor;
        }
    }

    public static float dot64(float[] v1, int offset1, float[] v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1, offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2, offset2);
        var multiplyResult = a.mul(b);
        return multiplyResult.reduceLanes(VectorOperators.ADD);
    }

    public static float dotProduct(float[] v1, float[] v2) {
        return dotProduct(v1, 0, v2, 0, v1.length);
    }

    public static float dotProduct(float[] v1, int v1offset, float[] v2, int v2offset, int length) {
        final VectorSpecies<Float> SPECIES = pickBestSpecies(length);

        //Do not remove this code. There's a jdk perf bug we hit if removed
        if (SPECIES != FloatVector.SPECIES_PREFERRED && logger.isTraceEnabled())
            logger.trace("Picked {} for length {}", SPECIES, length);

        FloatVector sum = FloatVector.zero(SPECIES);
        int vectorizedLength = SPECIES.loopBound(length);

        int v1limit = v1offset + vectorizedLength;
        int v2limit = v2offset + vectorizedLength;
        int inc = SPECIES.length();

        // Process the vectorized part
        for (; v1offset < v1limit && v2offset < v2limit; v1offset += inc, v2offset += inc) {
            FloatVector a = FloatVector.fromArray(SPECIES, v1, v1offset);
            FloatVector b = FloatVector.fromArray(SPECIES, v2, v2offset);
            sum = sum.add(a.mul(b));
        }

        float res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (; v1offset < v1limit && v2offset < v2limit; v1offset++, v2offset++) {
            res += v1[v1offset] * v2[v2offset];
        }

        return res;
    }

    public static int dotProduct(byte[] v1, byte[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        var sum = IntVector.zero(IntVector.SPECIES_256);
        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length);

        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1, i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2, i).castShape(IntVector.SPECIES_256, 0);
            sum = sum.add(a.mul(b));
        }

        int res = sum.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            res += v1[i] * v2[i];
        }

        return res;
    }

    public static float cosineSimilarity(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vaMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        var vbMagnitude = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length);
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2, i);
            vsum = vsum.add(a.mul(b));
            vaMagnitude = vaMagnitude.add(a.mul(a));
            vbMagnitude = vbMagnitude.add(b.mul(b));
        }

        float sum = vsum.reduceLanes(VectorOperators.ADD);
        float aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        float bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            sum += v1[i] * v2[i];
            aMagnitude += v1[i] * v1[i];
            bMagnitude += v2[i] * v2[i];
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    public static float cosineSimilarity(byte[] v1, byte[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vsum = IntVector.zero(IntVector.SPECIES_256);
        var vaMagnitude = IntVector.zero(IntVector.SPECIES_256);
        var vbMagnitude = IntVector.zero(IntVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length);
        // Process the vectorized part, convert from 8 bytes to 8 ints
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1, i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2, i).castShape(IntVector.SPECIES_256, 0);
            vsum = vsum.add(a.mul(b));
            vaMagnitude = vaMagnitude.add(a.mul(a));
            vbMagnitude = vbMagnitude.add(b.mul(b));
        }

        int sum = vsum.reduceLanes(VectorOperators.ADD);
        int aMagnitude = vaMagnitude.reduceLanes(VectorOperators.ADD);
        int bMagnitude = vbMagnitude.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            sum += v1[i] * v2[i];
            aMagnitude += v1[i] * v1[i];
            bMagnitude += v2[i] * v2[i];
        }

        return (float) (sum / Math.sqrt(aMagnitude * bMagnitude));
    }

    /**
     * Picks the best Species to use based on the vector length.
     *
     * This is useful for short vectors, since we avoid the tail processing
     * (as happens with PQ subvectors)
     */
    private static VectorSpecies<Float> pickBestSpecies(int length) {
        return switch (length) {
            case 2, 3 -> FloatVector.SPECIES_64;
            case 4, 5, 6, 7 -> FloatVector.SPECIES_128;
            case 8, 9, 10, 11, 12, 13, 14, 15 -> FloatVector.SPECIES_256;
            default -> FloatVector.SPECIES_PREFERRED;
        };
    }

    public static float squareDistance(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }
        VectorSpecies<Float> SPECIES = pickBestSpecies(v1.length);
        var vdiffSumSquared = FloatVector.zero(SPECIES);

        int vectorizedLength = SPECIES.loopBound(v1.length);
        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += SPECIES.length()) {
            var a = FloatVector.fromArray(SPECIES, v1, i);
            var b = FloatVector.fromArray(SPECIES, v2, i);

            var diff = a.sub(b);
            vdiffSumSquared = vdiffSumSquared.add(diff.mul(diff));
        }

        float diffSumSquared = vdiffSumSquared.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            diffSumSquared += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }

        return diffSumSquared;
    }

    public static int squareDistance(byte[] v1, byte[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        var vdiffSumSquared = IntVector.zero(IntVector.SPECIES_256);

        int vectorizedLength = ByteVector.SPECIES_64.loopBound(v1.length);
        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += ByteVector.SPECIES_64.length()) {
            var a = ByteVector.fromArray(ByteVector.SPECIES_64, v1, i).castShape(IntVector.SPECIES_256, 0);
            var b = ByteVector.fromArray(ByteVector.SPECIES_64, v2, i).castShape(IntVector.SPECIES_256, 0);

            var diff = a.sub(b);
            vdiffSumSquared = vdiffSumSquared.add(diff.mul(diff));
        }

        int diffSumSquared = vdiffSumSquared.reduceLanes(VectorOperators.ADD);

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            diffSumSquared += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        }

        return diffSumSquared;
    }

    public static void addInPlace(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = FloatVector.SPECIES_PREFERRED.loopBound(v1.length);

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2, i);
            a.add(b).intoArray(v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            v1[i] = v1[i] + v2[i];
        }
    }

    public static float[] sub(float[] lhs, float[] rhs) {
        if (lhs.length != rhs.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        float[] result = new float[lhs.length];
        int vectorizedLength = (lhs.length / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, lhs, i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, rhs, i);
            var subResult = a.sub(b);
            subResult.intoArray(result, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < lhs.length; i++) {
            result[i] = lhs[i] - rhs[i];
        }

        return result;
    }
}
