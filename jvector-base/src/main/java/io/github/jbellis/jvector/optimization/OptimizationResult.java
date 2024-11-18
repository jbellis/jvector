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

package io.github.jbellis.jvector.optimization;

public class OptimizationResult {
    /**
     * The solution of the optimization problem.
     */
    public final float[] x;

    /**
     * The error achieved by the solver.
     */
    final public double error;

    /**
     * The number of iterations performed by the solver.
     */
    final public int iterations;

    /**
     * The loss value achieved in the last iteration.
     */
    final public float lastLoss;

    /**
     * Constructs an OptimizationResult
     * @param x the solution of the optimization problem
     * @param error the error achieved by the solver
     */
    public OptimizationResult(float[] x, double error, int iterations, float lastLoss) {
        this.x = x;
        this.error = error;
        this.iterations = iterations;
        this.lastLoss = lastLoss;
    }
}
