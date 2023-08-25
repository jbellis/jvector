package com.github.jbellis.jvector.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.util.List;

public class SimdOps {
    public static float simdSum(float[] vector) {
        float sum = 0.0f;
        int vectorizedLength = (vector.length / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vector, i);
            sum += a.reduceLanes(VectorOperators.ADD);
        }

        // Process the tail
        for (int i = vectorizedLength; i < vector.length; i++) {
            sum += vector[i];
        }

        return sum;
    }

    public static float[] simdSum(List<float[]> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("Input list cannot be null or empty");
        }

        int dimension = vectors.get(0).length;
        float[] sum = new float[dimension];

        // Process each vector from the list
        for (float[] vector : vectors) {
            simdAddInPlace(sum, vector);
        }

        return sum;
    }

    /**
     * Divide v1 by v2, in place (v1 will be modified)
     */
    public static void simdDivInPlace(float[] vector, float divisor) {
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

    /**
     * Multiplies v1 by v2, in place (v1 will be modified)
     */
    public static void simdMulInPlace(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = (v1.length / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2, i);
            var multiplyResult = a.mul(b);
            multiplyResult.intoArray(v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            v1[i] = v1[i] * v2[i];
        }
    }

    /**
     * Computes the dot product of the first two floats in each vector
     * at the given offsets
     */
    public static float dot64(float[] v1, int offset1, float[] v2, int offset2) {
        var a = FloatVector.fromArray(FloatVector.SPECIES_64, v1, offset1);
        var b = FloatVector.fromArray(FloatVector.SPECIES_64, v2, offset2);
        var multiplyResult = a.mul(b);
        return multiplyResult.reduceLanes(VectorOperators.ADD);
    }

    /**
     * Adds v2 into v1, in place (v1 will be modified)
     */
    public static void simdAddInPlace(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have the same length");
        }

        int vectorizedLength = (v1.length / FloatVector.SPECIES_PREFERRED.length()) * FloatVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (int i = 0; i < vectorizedLength; i += FloatVector.SPECIES_PREFERRED.length()) {
            var a = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v1, i);
            var b = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, v2, i);
            var addResult = a.add(b);
            addResult.intoArray(v1, i);
        }

        // Process the tail
        for (int i = vectorizedLength; i < v1.length; i++) {
            v1[i] = v1[i] + v2[i];
        }
    }

    /**
     * @return lhs - rhs, element-wise
     */
    public static float[] simdSub(float[] lhs, float[] rhs) {
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
