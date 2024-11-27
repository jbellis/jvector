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

package io.github.jbellis.jvector.example;

import io.github.jbellis.jvector.util.MathUtil;

public class kumaraswamyApproximationScalar {
    static float inverseKumaraswamy(float value, float a, float b) {
        var temp = (float) Math.pow(1 - value, 1.f / b);   // (1 - v) ** (1 / b)
        return (float) Math.pow(1 - temp, 1.f / a);        // (1 - v) ** (1 / a)
    }

    static float inverseKumaraswamyExpLog(float value, float a, float b) {
        var temp = (float) Math.exp(Math.log(1 - value) / b);   // (1 - v) ** (1 / b)
        return (float) Math.exp(Math.log(1 - temp) / a);        // (1 - v) ** (1 / a)
    }

    static float inverseKumaraswamyExpLogApprox(float value, float a, float b) {
        var temp = MathUtil.fastExp(MathUtil.fastLog(1.f - value) / b);   // (1 - v) ** (1 / b)
        return MathUtil.fastExp(MathUtil.fastLog(1.f - temp) / a);        // (1 - v) ** (1 / a)
    }


    public static void testKumaraswamyApproximation(float a, float b) {
        int trials = 10_000_000;
        int errorTrials = 1_000_000;

        float v = 0.2f;

        long startTime, endTime;
        double duration;
        double error;
        float dummyAccum;

        System.out.println("a=" + a + " b=" + b);

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
        System.out.println("\tTraditional inverse Kumaraswamy took " + duration + " nanoseconds");
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
        System.out.println("\tExp/Log inverse Kumaraswamy took " + duration + " nanoseconds");

        error = 0f;
        for (int i = 0; i < errorTrials; i++) {
            float rv = (float) Math.random();
            error += Math.abs(inverseKumaraswamy(rv, a, b) - inverseKumaraswamyExpLog(rv, a, b));
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
        System.out.println("\tExp/Log approx inverse Kumaraswamy took " + duration + " nanoseconds");

        error = 0;
        for (int i = 0; i < errorTrials; i++) {
            float rv = (float) Math.random();
            error += Math.abs(inverseKumaraswamy(rv, a, b) - inverseKumaraswamyExpLogApprox(rv, a, b));
        }
        error /= trials;
        System.out.println("\tError " + error);

        System.out.println(dummyAccum);
    }

    public static void main(String[] args) {
        testKumaraswamyApproximation(1.3f, 1.15f);
        testKumaraswamyApproximation(0.5f, 0.3f);
        testKumaraswamyApproximation(1.f, 1.f);
    }
}
