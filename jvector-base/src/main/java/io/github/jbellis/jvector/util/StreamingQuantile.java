package io.github.jbellis.jvector.util;

import java.util.concurrent.ThreadLocalRandom;

/// Maintains a minimal data structure to estimate a target quantile from streaming data.
///
/// Implements a modified version of the Frugal-1U algorithm in the paper:
/// Ma, Muthukrishnan, Sandler
/// "Frugal Streaming for Estimating Quantiles: One (or two) memory suffices"
/// 2014
/// [...](https://arxiv.org/abs/1407.1121)
///
/// We found that a step of 1 / Math.sqrt(count) helps reduce the variance,
/// where count is the current number of insertions.
/// Note that the algorithm is sensitive to adversarial orderings of the data.
public class StreamingQuantile {
    public final ThreadLocalRandom randomGenerator = ThreadLocalRandom.current();

    private final double targetQuantile;
    private double quantileEstimate;
    private int count;

    /**
     * Constructor for the targeted quantile estimator
     * @param targetQuantile the quantile of interest
     */
    public StreamingQuantile(double targetQuantile) {
        if (targetQuantile < 0 || targetQuantile > 1) {
            throw new IllegalArgumentException("The target quantiles must be between 0 and 1 inclusive.");
        }
        this.targetQuantile = targetQuantile;
        this.quantileEstimate = 0;
        this.count = 0;
    }

    /**
     * Returns the approximate quantile
     */
    public double quantile() {
        return quantileEstimate;// * resolution;
    }

    public void insert(double value) {
        count++;
        double delta = 1 / Math.sqrt(count);

        var rand = randomGenerator.nextFloat();
        if (value > quantileEstimate && rand < targetQuantile) {
            quantileEstimate += delta;
        } else if (value < quantileEstimate && rand > targetQuantile) {
            quantileEstimate -= delta;
        }
    }

    public void reset() {
        this.quantileEstimate = 0;
        this.count = 0;
    }
}
