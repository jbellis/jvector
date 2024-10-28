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

import java.util.Arrays;
import java.util.stream.IntStream;

public abstract class LossFunction {
    final private int nDims;

    private double[] minBounds;
    private double[] maxBounds;

    public LossFunction(int nDims) {
        if (nDims <= 0) {
            throw new IllegalArgumentException("The standard deviation initSigma must be positive");
        }
        this.nDims = nDims;
    }

    // Does not perform projection
    public abstract double compute(double[] x);

    // Performs in-place projection
    public double projectCompute(double[] x) {
        project(x);
        return compute(x);
    }

    public void setMinBounds(double[] bounds) {
        if (nDims != bounds.length) {
            throw new IllegalArgumentException("The length of bounds should match the number of dimensions");
        }
        minBounds = bounds;
    }

    public double[] getMinBounds() {
        return minBounds;
    }

    public void setMaxBounds(double[] bounds) {
        if (nDims != bounds.length) {
            throw new IllegalArgumentException("The length of bounds should match the number of dimensions");
        }
        maxBounds = bounds;
    }

    public double[] getMaxBounds() {
        return maxBounds;
    }

    // projection
    public double[] project(double[] x, boolean inPlace) {
        double[] copy;
        if (inPlace) {
            copy = x;
        }
        else {
            copy = Arrays.copyOf(x, x.length);
        }
        IntStream.range(0, x.length).forEach(d -> copy[d] = Math.min(Math.max(x[d], minBounds[d]), maxBounds[d]));
        return copy;
    }

    // in-place projection
    public void project(double[] x) {
        project(x, true);
    }
}