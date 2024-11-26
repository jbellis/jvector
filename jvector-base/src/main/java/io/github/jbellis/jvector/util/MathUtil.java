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

package io.github.jbellis.jvector.util;

public class MathUtil {
    // looks silly at first but it really does make code more readable
    public static float square(float a) {
        return a * a;
    }

    // looks silly at first but it really does make code more readable
    public static double square(double a) {
        return a * a;
    }

    /*
     Fast exponential
     https://codingforspeed.com/using-faster-exponential-approximation/
     */
    public static float fastExp(float x) {
        x = 1.f + x / 1024;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x; x *= x; x *= x;
        x *= x; x *= x;
        return x;
    }

    /*
     Fast natural logarithm on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5
     https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
     */
    public static float fastLog(float x) {
        float m, r, s, t, i, f;
        int e;

        int temp = Float.floatToIntBits(x);
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
}
