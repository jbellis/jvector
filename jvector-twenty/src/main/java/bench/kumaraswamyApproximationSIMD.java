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

package bench;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

public class kumaraswamyApproximationSIMD {
    /*
     Vectorized fast exponential exp(x / c)
     https://codingforspeed.com/using-faster-exponential-approximation/
     */
    public static FloatVector fastExp(FloatVector x) {
        x = x.div(1024).add(const1f);
        x = x.mul(x); x = x.mul(x); x = x.mul(x); x = x.mul(x);
        x = x.mul(x); x = x.mul(x); x = x.mul(x); x = x.mul(x);
        x = x.mul(x);
        x = x.mul(x);
        return x;
    }

    private static FloatVector const119209290e7f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.19209290e-7f);
    private static IntVector const0x3f2aaaab = IntVector.broadcast(IntVector.SPECIES_PREFERRED, 0x3f2aaaab);
    private static IntVector const0xff800000 = IntVector.broadcast(IntVector.SPECIES_PREFERRED, 0xff800000);
    private static FloatVector const0230836749f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.230836749f);
    private static FloatVector const0279208571f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -0.279208571f);
    private static FloatVector const0331826031f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.331826031f);
    private static FloatVector const0498910338f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, -0.498910338f);
    private static FloatVector const0693147182f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 0.693147182f);
    private static FloatVector const1f = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f);

    /*
     Vectorized fast natural logarithm on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5.
     https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
     */
    public static FloatVector fastLogPreferred(FloatVector x) {
        IntVector temp = x.reinterpretAsInts();
        var e = temp.sub(const0x3f2aaaab).and(const0xff800000);
        var m = temp.sub(e).reinterpretAsFloats();
        var i = e.castShape(FloatVector.SPECIES_PREFERRED, 0).reinterpretAsFloats().mul(const119209290e7f);  // 0x1.0p-23

        /* m in [2/3, 4/3] */
        var f = m.sub(const1f);
        var s = f.mul(f);

        /* Compute log1p(f) for f in [-1/3, 1/3] */
        var r = f.fma(const0230836749f, const0279208571f);  // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        var t = f.fma(const0331826031f, const0498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2)
        r = r.fma(s, t);
        r = r.fma(s, f);
        r = i.fma(const0693147182f, r); // 0x1.62e430p-1 // log(2)
        return r;
    }

    static FloatVector inverseKumaraswamy(FloatVector vector, float a, float b) {
        FloatVector oneOverA = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f / a);
        FloatVector oneOverB = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f / b);
        var temp = const1f.sub(vector).pow(oneOverB);  // (1 - v) ** (1 / b)
        return const1f.sub(temp).pow(oneOverA);        // (1 - v) ** (1 / a)
    }

    static FloatVector inverseKumaraswamyExpLogApprox(FloatVector vector, float a, float b) {
        FloatVector oneOverA = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f / a);
        FloatVector oneOverB = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.f / b);
        var temp = fastExp(fastLogPreferred(const1f.sub(vector)).mul(oneOverB));  // (1 - v) ** (1 / b)
        return fastExp(fastLogPreferred(const1f.sub(temp)).mul(oneOverA));        // (1 - v) ** (1 / a)
    }

    public static void testKumaraswamyApproximation(float a, float b) {
        int trials = 10_000_000;
        int errorTrials = 1_000_000;

        int nDims = FloatVector.SPECIES_PREFERRED.length();

        float[] vArr = new float[nDims];
        for (int d = 0; d < nDims; d++) {
            vArr[d] = (float) Math.random();
        }
        FloatVector v = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vArr, 0);

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
            inverseKumaraswamy(v, a, b);
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / (trials * nDims);
        System.out.println("\tTraditional inverse Kumaraswamy took " + duration + " nanoseconds");
        //---------------------------------------------------

        //---------------------------------------------------
        // Exp/Log approx
        //---------------------------------------------------
        dummyAccum = 0;
        startTime = System.nanoTime();
        for (int i = 0; i < trials; i++) {
            inverseKumaraswamyExpLogApprox(v, a, b);
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / (trials * nDims);
        System.out.println("\tExp/Log approx inverse Kumaraswamy took " + duration + " nanoseconds");

        error = 0;
        for (int i = 0; i < errorTrials; i++) {
            float[] vArrTemp = new float[nDims];
            for (int d = 0; d < nDims; d++) {
                vArrTemp[d] = (float) Math.random();
            }
            FloatVector vTemp = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, vArrTemp, 0);

            FloatVector v1 = inverseKumaraswamy(vTemp, a, b);
            FloatVector v2 = inverseKumaraswamyExpLogApprox(vTemp, a, b);
            error += v1.sub(v2).abs().reduceLanes(VectorOperators.ADD);
        }
        error /= trials * nDims;
        System.out.println("\tError " + error);

        System.out.println(dummyAccum);
    }

    public static void main(String[] args) {
        testKumaraswamyApproximation(1.3f, 1.15f);
        testKumaraswamyApproximation(0.5f, 0.3f);
        testKumaraswamyApproximation(1.f, 1.f);
    }
}
