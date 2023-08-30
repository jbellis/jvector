package com.github.jbellis.jvector.vector;

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
}
