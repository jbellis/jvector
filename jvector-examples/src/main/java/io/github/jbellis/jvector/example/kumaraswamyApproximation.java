package io.github.jbellis.jvector.example;


public class kumaraswamyApproximation {
    static float exp1024(float x) {
        x = 1.f + x / 1024;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x;
        return x;
    }
    /* natural log on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5 */
    static float logApprox(float a)
    {
        float m, r, s, t, i, f;
        int e;

        int temp = Float.floatToIntBits(a);
        e = (temp - 0x3f2aaaab) & 0xff800000;
        m = Float.intBitsToFloat(temp - e);
        i = (float)e * 1.19209290e-7f; // 0x1.0p-23
        /* m in [2/3, 4/3] */
        f = m - 1.0f;
        s = f * f;
        /* Compute log1p(f) for f in [-1/3, 1/3] */
        r = Math.fma(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        t = Math.fma(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
        r = Math.fma(r, s, t);
        r = Math.fma(r, s, f);
        r = Math.fma(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2)
        return r;
    }

    static float inverseKumaraswamy(float value, float a, float b) {
        var temp = (float) Math.pow(1 - value, 1.f / b);   // (1 - v) ** (1 / b)
        return (float) Math.pow(1 - temp, 1.f / a);        // (1 - v) ** (1 / a)
    }

    static float inverseKumaraswamyExpLog(float value, float a, float b) {
        var temp = (float) Math.exp(Math.log(1 - value) / b);   // (1 - v) ** (1 / b)
        return (float) Math.exp(Math.log(1 - temp) / a);        // (1 - v) ** (1 / a)
    }

    static float inverseKumaraswamyExpLogApprox(float value, float a, float b) {
        var temp = exp1024(logApprox(1.f - value) / b);   // (1 - v) ** (1 / b)
        return exp1024(logApprox(1.f - temp) / a);        // (1 - v) ** (1 / a)
    }


    public static void main(String[] args) {
        int trials = 100_000_000;
        int errorTrials = 1_000_000;

        float a = 1.15f;
        float b = 0.7f;
        float v = 0.2f;

        long startTime, endTime;
        double duration;
        double error;
        float dummyAccum;

        //---------------------------------------------------
        // Original
        //---------------------------------------------------
        dummyAccum = 0;
        startTime = System.nanoTime();
        for (int i = 0; i < trials; i++) {
            dummyAccum += inverseKumaraswamy(v, a, b);
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / trials;
        System.out.println("Traditional inverse Kumaraswamy took " + duration + " nanoseconds");
        System.out.println(dummyAccum / trials);
        //---------------------------------------------------

        //---------------------------------------------------
        // Exp/Log
        //---------------------------------------------------
        dummyAccum = 0;
        startTime = System.nanoTime();
        for (int i = 0; i < trials; i++) {
            dummyAccum += inverseKumaraswamyExpLog(v, a, b);
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / trials;
        System.out.println("Exp/Log inverse Kumaraswamy took " + duration + " nanoseconds");
        System.out.println(dummyAccum / trials);

        error = 0f;
        for (int i = 0; i < errorTrials; i++) {
            float rv = (float) Math.random();
            error += inverseKumaraswamy(rv, a, b) - inverseKumaraswamyExpLog(rv, a, b);
        }
        error /= trials;
        System.out.println("\tError " + error);
        //---------------------------------------------------

        //---------------------------------------------------
        // Exp/Log approx
        //---------------------------------------------------
        dummyAccum = 0;
        startTime = System.nanoTime();
        for (int i = 0; i < trials; i++) {
            dummyAccum += inverseKumaraswamyExpLogApprox(v, a, b);
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / trials;
        System.out.println("Exp/Log approx inverse Kumaraswamy took " + duration + " nanoseconds");
        System.out.println(dummyAccum / trials);

        error = 0;
        for (int i = 0; i < errorTrials; i++) {
            float rv = (float) Math.random();
            error += inverseKumaraswamy(rv, a, b) - inverseKumaraswamyExpLogApprox(rv, a, b);
        }
        error /= trials;
        System.out.println("\tError " + error);

//        float exact = (float) Math.log(v);
//        System.out.println(exact);
//        float approx = my_faster_logf(v);
//        System.out.println(approx);
    }
}
