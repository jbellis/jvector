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
 * Added a modification to support box constraints (min/max values for each parameter) by projecting onto
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
    private double lrMu = 1;

    // learning rate for the standard deviation sigma (i.e., the solution)
    // This value can never be zero. Using zero to indicate that it has not been initialized.
    private double lrSigma = 0;

    // value of the stopping condition for the optimization
    private double tol = 1e-6;

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
    public void setLrMu(double lrMu) {
        this.lrMu = lrMu;
    }

    /**
     * Sets the learning rate used to update the variance of the solution throughout the optimization process.
     * Used to override the default number (see Table 1 in [1]).
     * @param lrSigma
     */
    public void setLrSigma(double lrSigma) {
        this.lrSigma = lrSigma;
    }

    /**
     * Internal method used to compute the default learning rate used update the variance of the solution.
     *
     * @param nDims the number of parameters of the optimization problem.
     * @return the learning rate
     */
    private double computeLrSigma(int nDims) {
        if (lrSigma == 0) {
            return (9 + 3 * Math.log(nDims)) / (5 * nDims * Math.sqrt(nDims));
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
    public void setTol(double tol) {
        this.tol = tol;
    }

    /**
     * NES utilizes rank-based fitness shaping in order to render the algorithm invariant under
     * monotonically increasing (i.e., rank preserving) transformations of the loss function.
     * For this, the samples losses are transformed into a set of utility values using the method in Section 3.1 of [1].
     *
     * @param lossFun the loss function
     * @param samples the set of samples at which to evaluate the loss function
     * @return the utilities corresponding to each sample
     */
    private float[] computeUtilities(LossFunction lossFun, double[][] samples) {
        // See section 3.1 in https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        var us = IntStream.range(1, samples.length + 1).asDoubleStream().map(i -> Math.max(0., Math.log(1 + (double) samples.length / 2) - Math.log(i))).toArray();
        var uSum = Arrays.stream(us).sum();
        final var utilities = Arrays.stream(us).map(u -> (u / uSum) - (1. / samples.length)).toArray();

        var funValues = Arrays.stream(samples).mapToDouble(lossFun::projectCompute).toArray();

        var indices = IntStream.range(0, samples.length).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, (i1, i2) -> -1 * Double.compare(funValues[i1], funValues[i2]));

        float[] sortedUtilities = new float[samples.length];
        IntStream.range(0, samples.length).forEach(i -> sortedUtilities[indices[i]] = (float) utilities[i]);
        return sortedUtilities;
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for the following parameters distributions
     * (1) separable and normal (Algorithm 6 in [1]).
     * (2) multinormal (Algorithm 5 in [1]). The latter is not implemented yet.
     * We extended the algorithm to support box constraints (min/max values for each parameter) by projecting onto
     * the feasible set.
     *
     * @param lossFun the loss function, that allows to specify its own feasible set (box constraints).
     * @param initialSolution The initial solution.
     * @return the optimization result
     */
    public OptimizationResult optimize(LossFunction lossFun, double[] initialSolution) {
        return optimize(lossFun, initialSolution, 0.5);
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for the following parameters distributions
     * (1) separable and normal (Algorithm 6 in [1]).
     * (2) multinormal (Algorithm 5 in [1]). The latter is not implemented yet.
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
    public OptimizationResult optimize(LossFunction lossFun, double[] initialSolution, double initSigma) {
        return optimizeSeparable(lossFun, initialSolution, initSigma);
    }

    /**
     * Runs the Exponential Natural Evolution Strategies (xNES) solver for a separable and normal parameter
     * distribution (Algorithm 6 in [1]) with a modification to support box constraints (min/max) by projecting onto
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
    private OptimizationResult optimizeSeparable(LossFunction lossFun, double[] initialSolution, double initSigma) {
        if (initSigma <= 0) {
            throw new IllegalArgumentException("The standard deviation initSigma must be positive");
        }

        var random = ThreadLocalRandom.current();

        var nDims = initialSolution.length;

        var nSamples = computeNSamples(nDims);
        var lrSigma = computeLrSigma(nDims);

        // Initialize mu and sigma
        var mu = new double[nDims];
        var sigma = new double[nDims];
        for (int d = 0; d < nDims; d++) {
            mu[d] = initialSolution[d];
            sigma[d] = initSigma;
        }

        // create their natural gradient
        var deltaMu = new double[nDims];
        var deltaSigma = new double[nDims];

        var oldFunVal = lossFun.compute(mu);

//        int iter = 0;
        double error = tol + 1.;
        while (error > tol) {
            // generate samples used to compute the natural gradient
            double[][] rawSamples = new double[nSamples][];
            double[][] samples = new double[nSamples][];
            for (int i = 0; i < nSamples; i++) {
                rawSamples[i] = new double[nDims];
                samples[i] = new double[nDims];

                for (int d = 0; d < nDims; d++) {
                    var v = random.nextGaussian();
                    rawSamples[i][d] = v;
                    var z = mu[d] + sigma[d] * v;
                    samples[i][d] = z;
                }
            }

            // See section 3.1 in [1].
            var utilities = computeUtilities(lossFun, samples);

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
                mu[d] = mu[d] + lrMu * sigma[d] * deltaMu[d];

                // Update the standard deviation in each dimension
                sigma[d] = sigma[d] * Math.exp(deltaSigma[d] * lrSigma / 2);
            }
            lossFun.project(mu);

            // Compute stopping criterion
            var newFunVal = lossFun.compute(mu);
            error = Math.abs(newFunVal - oldFunVal);
            oldFunVal = newFunVal;

//            String str = String.format("%d -> The solution is %.8f  %.8f with error %e", iter, mu[0], mu[1], error);
//            System.out.println(str);
//            iter += 1;
        }

        return new OptimizationResult(mu, error);
    }
}

