package com.github.jbellis.jvector.vector;

import java.util.concurrent.ThreadLocalRandom;

import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import com.github.jbellis.jvector.graph.GraphIndexTestCase;

public class TestVectorizationProvider {
    static boolean hasSimd = VectorizationProvider.vectorModulePresentAndReadable();

    @Test
    public void testDotProductFloat() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorizationProvider b = VectorizationProvider.getInstance();

        for (int i = 0; i < 1000; i++) {
            float[] v1 = GraphIndexTestCase.randomVector(ThreadLocalRandom.current(), 1021); //prime numbers
            float[] v2 = GraphIndexTestCase.randomVector(ThreadLocalRandom.current(), 1021); //prime numbers

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1,v2), b.getVectorUtilSupport().dotProduct(v1, v2), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1,v2), b.getVectorUtilSupport().cosine(v1, v2), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1,v2), b.getVectorUtilSupport().squareDistance(v1, v2), 0.00001f);
        }
    }

    @Test
    public void testDotProductByte() {
        Assume.assumeTrue(hasSimd);

        VectorizationProvider a = new DefaultVectorizationProvider();
        VectorizationProvider b = VectorizationProvider.getInstance();

        for (int i = 0; i < 1000; i++) {
            byte[] v1 = GraphIndexTestCase.randomVector8(ThreadLocalRandom.current(), 1021); //prime numbers
            byte[] v2 = GraphIndexTestCase.randomVector8(ThreadLocalRandom.current(), 1021); //prime numbers

            Assert.assertEquals(a.getVectorUtilSupport().dotProduct(v1,v2), b.getVectorUtilSupport().dotProduct(v1, v2));
            Assert.assertEquals(a.getVectorUtilSupport().cosine(v1,v2), b.getVectorUtilSupport().cosine(v1, v2), 0.00001f);
            Assert.assertEquals(a.getVectorUtilSupport().squareDistance(v1,v2), b.getVectorUtilSupport().squareDistance(v1, v2));
        }
    }
}
