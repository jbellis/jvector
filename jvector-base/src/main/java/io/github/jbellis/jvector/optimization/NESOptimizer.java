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
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Implements Exponential Natural Evolution Strategies (xNES) for the separable case (Algorithm 6 in [1]).
 * Added a simple modification to support box constraints (min/max) by projecting onto the feasible set.
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
     * Constructor
     * @param dist The parameter distribution to be used for the optimization.
     *             Currently, only SEPARABLE is supported.
     */
    private NESOptimizer(Distribution dist) {
        if (dist != Distribution.SEPARABLE) {
            throw new UnsupportedOperationException("The multinormal case is not implemented yet.");
        }
        //this.distribution = dist;
    }

    public void setnSamples(int nSamples) {
        this.nSamples = nSamples;
    }

    public int getnSamples() {
        return nSamples;
    }

    private int computeNSamples(int nDims) {
        if (nSamples == 0) {
            return 4 + (int) Math.floor(3 * Math.log(nDims));
        } else {
            return nSamples;
        }
    }

    public void setLrMu(double lrMu) {
        this.lrMu = lrMu;
    }

    public void setLrSigma(double lrSigma) {
        this.lrSigma = lrSigma;
    }

    private double computeLrSigma(int nDims) {
        if (lrSigma == 0) {
            return ((9 + 3 * Math.log(nDims)) / (5 * nDims * Math.sqrt(nDims)));
        } else {
            return lrSigma;
        }
    }

    public void setTol(double tol) {
        this.tol = tol;
    }

    private float[] computeUtilities(LossFunction lossFun, double[][] samples) {
        // See section 3.1 in https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        var us = IntStream.range(1, samples.length + 1).asDoubleStream().map(i -> Math.max(0., Math.log(1 + (double) samples.length / 2) - Math.log(i))).toArray();
        var u_sum = Arrays.stream(us).sum();
        final var utilities = Arrays.stream(us).map(u -> (u / u_sum) - (1. / samples.length)).toArray();

        var funValues = Arrays.stream(samples).mapToDouble(lossFun::projectCompute).toArray();

        var indices = IntStream.range(0, samples.length).boxed().toArray(Integer[]::new);
        Arrays.sort(indices, (i1, i2) -> -1 * Double.compare(funValues[i1], funValues[i2]));

        float[] sorted_utilities = new float[samples.length];
        IntStream.range(0, samples.length).forEach(i -> sorted_utilities[indices[i]] = (float) utilities[i]);
        return sorted_utilities;
    }

    public double[] optimize(LossFunction lossFun, double[] initialSolution) {
        return optimize(lossFun, initialSolution, 0.5);
    }

    public double[] optimize(LossFunction lossFun, double[] initialSolution, double initSigma) {
        return optimizeSeparable(lossFun, initialSolution, initSigma);
    }

    private double[] optimizeSeparable(LossFunction lossFun, double[] initialSolution, double initSigma) {
        if (initSigma <= 0) {
            throw new IllegalArgumentException("The standard deviation initSigma must be positive");
        }

        var random = ThreadLocalRandom.current();

        var nDims = initialSolution.length;

        var _nSamples = computeNSamples(nDims);
        var _lrSigma = computeLrSigma(nDims);

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

        int iter = 0;
        double error = tol + 1.;
        while (error > tol) {
            double[][] rawSamples = new double[_nSamples][];
            double[][] samples = new double[_nSamples][];
            for (int i = 0; i < _nSamples; i++) {
                rawSamples[i] = new double[nDims];
                samples[i] = new double[nDims];

                for (int d = 0; d < nDims; d++) {
                    var v = random.nextGaussian();
                    rawSamples[i][d] = v;
                    var z = mu[d] + sigma[d] * v;
                    samples[i][d] = z;
                }
            }

            var utilities = computeUtilities(lossFun, samples);

            // Compute gradients:
            for (int d = 0; d < nDims; d++) {
                deltaMu[d] = 0;
                deltaSigma[d] = 0;
                for (int i = 0; i < _nSamples; i++) {
                    deltaMu[d] += utilities[i] * rawSamples[i][d];
                    deltaSigma[d] += utilities[i] * (MathUtil.square(rawSamples[i][d]) - 1);
                }
            }

            for (int d = 0; d < nDims; d++) {
                // Update mean in each dimension
                mu[d] = mu[d] + lrMu * sigma[d] * deltaMu[d];

                // Update the standard deviation in each dimension
                sigma[d] = sigma[d] * Math.exp(deltaSigma[d] * _lrSigma / 2);
            }
            lossFun.project(mu);

            var newFunVal = lossFun.compute(mu);
            error = Math.abs(newFunVal - oldFunVal);
            oldFunVal = newFunVal;

            String str = String.format("%d -> The solution is %.8f  %.8f with error %e", iter, mu[0], mu[1], error);
            System.out.println(str);
            iter += 1;
        }

        return mu;
    }

    public static void main(String[] args) {
        class TestLossFunction extends LossFunction {
            public TestLossFunction(int nDims) {
                super(nDims);
            }

            public double compute(double[] x) {
                return -1 * DoubleStream.of(x).map(MathUtil::square).sum();
            }
        }
        var loss = new TestLossFunction(2);
        loss.setMinBounds(new double[] {-1000, -1000});
        loss.setMaxBounds(new double[] {1000, 1000});

        double[] initialSolution = {1, 1};
        var xnes = new NESOptimizer(Distribution.SEPARABLE);
        xnes.setTol(1e-9);
        var sol = xnes.optimize(loss, initialSolution, 0.5);
        String str = String.format("The solution is %.8f  %.8f", sol[0], sol[1]);
        System.out.println(str);
    }
}

