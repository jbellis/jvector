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

import io.github.jbellis.jvector.util.MathUtil;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

/**
 * Implements Exponential Natural Evolution Strategies (xNES) for the separable case (Algorithm 6 in [1]).
 * It implements loss maximization.
 * We added a modification to support box constraints (min/max values for each parameter) by projecting onto
 * the feasible set.
 * <p>
 * [1]
 * Wierstra, Schaul, Glasmachers, Sun, Peters, and Schmidhuber
 * Natural Evolution Strategies
 * <a href="https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf">Link</a>
 */
public class NESOptimizer {
    public enum Distribution {
        MULTINORMAL, // This is for adding future support for the multinormal case (Algorithm 5 in [1])
        SEPARABLE
    }

    // number of samples to estimate the natural gradient
    // This value can never be zero. Using zero to indicate that it has not been initialized.
    private int nSamples = 0; // using 0 to denote uni

    // learning rate for the mean mu (i.e., the solution)
    private float lrMu = 1;

    // learning rate for the standard deviation sigma (i.e., the solution)
    // This value can never be zero. Using zero to indicate that it has not been initialized.
    private float lrSigma = 0;

    // value of the stopping condition for the optimization
    private float tol = 1e-6f;

    // Maximum number of iterations performed by the solver
    // Using zero to indicate that maxIterations is infinite
    private int maxIterations = 0;

    // the distribution to use
    // private final Distribution distribution;

    /**
     * Constructs a NESOptimizer with the specified distribution.
     *
     * @param dist The parameter distribution to be used for the optimization.
     *             Currently, only SEPARABLE is supported.
     */
    public NESOptimizer(Distribution dist) {
        if (dist != Distribution.SEPARABLE) {
            throw new UnsupportedOperationException("The multinormal case is not implemented yet.");
        }
        //this.distribution = dist;
    }

    /**
     * Sets the number of samples used for the natural gradient.
     * Used to override the default number (see Table 1 in [1]).
     *
     * @param nSamples the number of samples
     */
    public void setNSamples(int nSamples) {
        this.nSamples = nSamples;
    }

    /**
     * Internal method used to compute the default number of samples used for the natural gradient.
     * @param nDims the number of parameters of the optimization problem.
     * @return the number of samples
     */
    private int computeNSamples(int nDims) {
        if (nSamples == 0) {
            return 2 * (4 + (int) Math.floor(3 * Math.log(nDims)));
        } else {
            return nSamples;
        }
    }

    /**
     * Sets the learning rate used to update the solution throughout the optimization process.
     * Used to override the default value of 1.
     *
     * @param lrMu the learning rate
     */
    public void setLrMu(float lrMu) {
        this.lrMu = lrMu;
    }

    /**
     * Sets the learning rate used to update the variance of the solution throughout the optimization process.
     * Used to override the default number (see Table 1 in [1]).
     * @param lrSigma the learning rate
     */
    public void setLrSigma(float lrSigma) {
        this.lrSigma = lrSigma;
    }

    /**
     * Internal method used to compute the default learning rate used update the variance of the solution.
     *
     * @param nDims the number of parameters of the optimization problem.
     * @return the learning rate
     */
    private float computeLrSigma(int nDims) {
        if (lrSigma == 0) {
            return (float) ((9 + 3 * Math.log(nDims)) / (5 * nDims * Math.sqrt(nDims)));
        } else {
            return lrSigma;
        }
    }

    /**
     * Sets the tolerance for the stopping criterion. The error measures that absolute value of the difference
     * of two evaluations of the loss function in consecutive iterations of the NES solver.
     * When the error is below the tolerance, the solver breaks the iterations.
     *
     * @param tol the maximum acceptable tolerance
     */
    public void setTol(float tol) {
        this.tol = tol;
    }

    /**
     * Sets the maximum number of iterations performed by the solver. Use zero to indicate an unbounded number
     * of interations.
     *
     * @param maxIterations the maximum number of iterations
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    /**
     * NES utilizes rank-based fitness shaping in order to render the algorithm invariant under
     * monotonically increasing (i.e., rank preserving) transformations of the loss function.
     * For this, the samples losses are transformed into a set of utility values using the method in Section 3.1 of [1].
     *
     * @param lossFun the loss function
     * @param samples the set of samples at which to evaluate the loss function
     * @param utilities the utility corresponding to each sample
     */
    private float[] computeUtilities(LossFunction lossFun, float[][] samples, float[] utilities) {
        // See section 3.1 in https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        var us = IntStream.range(1, samples.length + 1).asDoubleStream().map(i -> Math.max(0., Math.log(1 + (double) samples.length / 2) - Math.log(i))).toArray();
        var uSum = Arrays.stream(us).sum();
        final var tempUtilities = Arrays.stream(us).map(u -> (u / uSum) - (1. / samples.length)).toArray();

        var funValues = Arrays.stream(samples).mapToDouble(lossFun::projectCompute).toArray();

        var indices = IntStream.range(0, samples.length).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, (i1, i2) -> -1 * Double.compare(funValues[i1], funValues[i2]));

        IntStream.range(0, samples.length).forEach(i -> utilities[indices[i]] = (float) tempUtilities[i]);
        return utilities;
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for the following parameters distributions
     * (1) separable and normal (Algorithm 6 in [1]).
     * (2) multinormal (Algorithm 5 in [1]). The latter is not implemented yet.
     * It implements loss maximization.
     * We extended the algorithm to support box constraints (min/max values for each parameter) by projecting onto
     * the feasible set.
     *
     * @param lossFun the loss function, that allows to specify its own feasible set (box constraints).
     * @param initialSolution The initial solution.
     * @return the optimization result
     */
    public OptimizationResult optimize(LossFunction lossFun, float[] initialSolution) {
        return optimize(lossFun, initialSolution, 0.5f);
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for the following parameters distributions
     * (1) separable and normal (Algorithm 6 in [1]).
     * (2) multinormal (Algorithm 5 in [1]). The latter is not implemented yet.
     * It implements loss maximization.
     * We extended the algorithm to support box constraints (min/max values for each parameter) by projecting onto
     * the feasible set.
     *
     * @param lossFun the loss function, that allows to specify its own feasible set (box constraints).
     * @param initialSolution the initial solution.
     * @param initSigma the initial variance of the sampling method used to generate the natural gradient.
     *                  The larger its value, the more likely we will avoid local minima.
     *                  However, a value too large might yield very poor a poor natural gradient.
     *                  Set with care depending on the problem although a value in [0.5, 1] is reasonable.
     * @return the optimization result
     */
    public OptimizationResult optimize(LossFunction lossFun, float[] initialSolution, float initSigma) {
        return optimizeSeparable(lossFun, initialSolution, initSigma);
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for a separable and normal parameter
     * distribution (Algorithm 6 in [1]) with a modification to support box constraints (min/max) by projecting onto
     * the feasible set. It implements loss maximization.
     *
     * @param lossFun the loss function, that allows to specify its own feasible set (box constraints).
     * @param initialSolution the initial solution.
     * @param initSigma the initial variance of the sampling method used to generate the natural gradient.
     *                  The larger its value, the more likely we will avoid local minima.
     *                  However, a value too large might yield very poor a poor natural gradient.
     *                  Set with care depending on the problem although a value in [0.5, 1] is reasonable.
     * @return the optimization result
     */
    private OptimizationResult optimizeSeparable(LossFunction lossFun, float[] initialSolution, float initSigma) {
        if (initSigma <= 0) {
            throw new IllegalArgumentException("The standard deviation initSigma must be positive");
        }

        var random = ThreadLocalRandom.current();

        var nDims = initialSolution.length;

        var nSamples = computeNSamples(nDims);
        var lrSigma = computeLrSigma(nDims);

        // Initialize mu and sigma
        var mu = new float[nDims];
        var sigma = new float[nDims];
        for (int d = 0; d < nDims; d++) {
            mu[d] = initialSolution[d];
            sigma[d] = initSigma;
        }

        // create their natural gradient
        var deltaMu = new float[nDims];
        var deltaSigma = new float[nDims];

        var oldFunVal = lossFun.compute(mu);

        float[][] rawSamples = new float[nSamples][];
        float[][] samples = new float[nSamples][];
        for (int i = 0; i < nSamples; i++) {
            rawSamples[i] = new float[nDims];
            samples[i] = new float[nDims];
        }

        float[] utilities = new float[nSamples];

        int iter = 0;
        double error = tol + 1.;
        while (error > tol && (maxIterations == 0 || iter < maxIterations)) {
            iter += 1;

            // generate samples used to compute the natural gradient
            for (int i = 0; i < nSamples; i++) {
                for (int d = 0; d < nDims; d++) {
                    var v = random.nextGaussian();
                    rawSamples[i][d] = (float) v;
                    var z = mu[d] + sigma[d] * v;
                    samples[i][d] = (float) z;
                }
            }

            // See section 3.1 in [1].
            computeUtilities(lossFun, samples, utilities);

            // Compute gradients:
            for (int d = 0; d < nDims; d++) {
                deltaMu[d] = 0;
                deltaSigma[d] = 0;
                for (int i = 0; i < nSamples; i++) {
                    deltaMu[d] += utilities[i] * rawSamples[i][d];
                    deltaSigma[d] += utilities[i] * (MathUtil.square(rawSamples[i][d]) - 1);
                }
            }

            for (int d = 0; d < nDims; d++) {
                // Update mean in each dimension
                mu[d] += lrMu * sigma[d] * deltaMu[d];

                // Update the standard deviation in each dimension
                sigma[d] *= (float) Math.exp(deltaSigma[d] * lrSigma / 2);
            }
            lossFun.project(mu);

            // Compute stopping criterion
            var newFunVal = lossFun.compute(mu);
            error = Math.abs(newFunVal - oldFunVal) / Math.abs(oldFunVal);
            oldFunVal = newFunVal;
        }

//        long endTime = System.nanoTime();
//        long duration = (endTime - startTime) / 1_000_000;  // in milliseconds.
//        System.out.println("Optimization time: " + duration + "ms " + iter);
//        System.out.println("Initial and Final sol: " + lossFun.compute(new float[]{1, 1}) + " " + oldFunVal);

        return new OptimizationResult(mu, error);
    }
}

