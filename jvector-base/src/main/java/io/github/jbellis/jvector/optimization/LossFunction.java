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

/**
 * Class that models a loss function that maps a multidimensional vector onto a real number.
 */
public abstract class LossFunction {
    // The number of dimensions
    final private int nDims;

    // The box constraints that define the feasible set.
    private float[] minBounds;
    private float[] maxBounds;

    /**
     * Constructs a LossFunction acting on vectors of the specified number of dimensions.
     * @param nDims the number of dimensions
     */
    public LossFunction(int nDims) {
        if (nDims <= 0) {
            throw new IllegalArgumentException("The standard deviation initSigma must be positive");
        }
        minBounds = new float[nDims];
        maxBounds = new float[nDims];
        for (int d = 0; d < nDims; d++) {
            minBounds[d] = Float.NEGATIVE_INFINITY;
            maxBounds[d] = Float.POSITIVE_INFINITY;
        }

        this.nDims = nDims;
    }

    /**
     * Computes the loss function. It assumes that input is within the feasible set
     * @param x the input vector
     * @return the loss
     */
    public abstract float compute(float[] x);

    /**
     * Computes the loss function and projects the input in-place onto the feasible set
     * @param x the input vector
     * @return the loss
     */
    public float projectCompute(float[] x) {
        project(x);
        return compute(x);
    }

    /**
     * Sets the minimum values of the box constraints.
     * @param bounds the specified minimum bound
     */
    public void setMinBounds(float[] bounds) {
        if (nDims != bounds.length) {
            throw new IllegalArgumentException("The length of bounds should match the number of dimensions");
        }
        minBounds = bounds;
    }

    /**
     * Gets the minimum values of the box constraints.
     * @return the minimum bound
     */
    public float[] getMinBounds() {
        return minBounds;
    }

    /**
     * Sets the maximum values of the box constraints.
     * @param bounds the specified maximum bound
     */
    public void setMaxBounds(float[] bounds) {
        if (nDims != bounds.length) {
            throw new IllegalArgumentException("The length of bounds should match the number of dimensions");
        }
        maxBounds = bounds;
    }

    /**
     * Gets the maximum values of the box constraints.
     * @return the maximum bound
     */
    public float[] getMaxBounds() {
        return maxBounds;
    }

    /**
     * Projects the input onto the feasible set. If in-place, the input array is modified;
     * otherwise, a copy is created and then projected.
     * @param x the input vector
     * @param inPlace If true, the input array is modified; otherwise, a copy is created and then projected.
     * @return the projected vector
     */
    public float[] project(float[] x, boolean inPlace) {
        float[] copy;
        if (inPlace) {
            copy = x;
        }
        else {
            copy = Arrays.copyOf(x, x.length);
        }
        IntStream.range(0, x.length).forEach(d -> copy[d] = Math.min(Math.max(x[d], minBounds[d]), maxBounds[d]));
        return copy;
    }

    /**
     * Projects the input in-place onto the feasible set.
     * @param x the input vector
     */
    public void project(float[] x) {
        project(x, true);
    }
}