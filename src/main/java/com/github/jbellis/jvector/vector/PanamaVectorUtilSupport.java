package com.github.jbellis.jvector.vector;

import java.util.List;

final class PanamaVectorUtilSupport implements VectorUtilSupport {

    @Override
    public float dotProduct(float[] a, float[] b)
    {
        return SimdOps.dotProduct(a, b);
    }

    @Override
    public float cosine(float[] v1, float[] v2)
    {
        return SimdOps.cosineSimilarity(v1, v2);
    }

    @Override
    public float squareDistance(float[] a, float[] b)
    {
        return SimdOps.squareDistance(a, b);
    }

    @Override
    public int dotProduct(byte[] a, byte[] b)
    {
        return SimdOps.dotProduct(a, b);
    }

    @Override
    public float cosine(byte[] a, byte[] b)
    {
        return SimdOps.cosineSimilarity(a, b);
    }

    @Override
    public int squareDistance(byte[] a, byte[] b)
    {
        return SimdOps.squareDistance(a, b);
    }

    @Override
    public float[] sum(List<float[]> vectors)
    {
        return SimdOps.sum(vectors);
    }

    @Override
    public float sum(float[] vector)
    {
        return SimdOps.sum(vector);
    }

    @Override
    public void divInPlace(float[] vector, float divisor)
    {
        SimdOps.divInPlace(vector, divisor);
    }

    @Override
    public float dot64(float[] v1, int offset1, float[] v2, int offset2) {
        return SimdOps.dot64(v1, offset1, v2, offset2);
    }

    @Override
    public void addInPlace(float[] v1, float[] v2) {
        SimdOps.addInPlace(v1, v2);
    }

    @Override
    public float[] sub(float[] lhs, float[] rhs) {
        return SimdOps.sub(lhs, rhs);
    }
}
