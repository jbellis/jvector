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

import io.github.jbellis.jvector.util.MathUtil;

public class kumaraswamyApproximationScalarAt0 {
    public static void approx_at0() {
        System.out.println(MathUtil.fastLog(0));
        System.out.println(MathUtil.fastExp(MathUtil.fastLog(0)));
        for (float c = 0.1f; c < 1.13f; c += 0.01f) {
            System.out.println("[" + c + ", " + MathUtil.fastExp(c * MathUtil.fastLog(0)) + "],");
        }
    }

    public static void main(String[] args) {
        approx_at0();
    }
}
