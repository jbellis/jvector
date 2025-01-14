package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.LuceneTestCase;
import org.apache.commons.math3.stat.StatUtils;
import org.junit.Test;

import java.util.Random;

public class TestStreamingQuantile extends LuceneTestCase {
    private enum RandomStreamCreator {
        GAUSSIAN {
            @Override
            public double[] createStream(Random random, int nSamples) {
                var vec = new double[nSamples];
                for (int i = 0; i < nSamples; i++) {
                    vec[i] = random.nextGaussian() + 1;
                }
                return vec;
            }
        },
        UNIFORM {
            @Override
            public double[] createStream(Random random, int nSamples) {
                var vec = new double[nSamples];
                for (int i = 0; i < nSamples; i++) {
                    vec[i] = random.nextFloat();
                }
                return vec;
            }
        };

        public abstract double[] createStream(Random random, int nSamples);
    }

    private void testQuantileEstimator(RandomStreamCreator streamCreator, int nTrials, int nSamples, int percentile,
                                       double errorThreshold) {
        var sq = new StreamingQuantile((double) percentile / 100);

        double averageRelativeError = 0;
        for (var trial = 0; trial < nTrials; trial++) {
            var stream = streamCreator.createStream(getRandom(), nSamples);

            sq.reset();
            for (var v : stream) {
                sq.insert(v);
            }
            double quantileEstimate = sq.quantile();

            double quantileTrue = StatUtils.percentile(stream, percentile);

            var relativeError = Math.abs(quantileTrue - quantileEstimate) / Math.abs(quantileTrue);
            averageRelativeError += relativeError;
        }
        averageRelativeError /= nTrials;
        assert averageRelativeError <= errorThreshold : String.format("%s > %s", averageRelativeError, errorThreshold);
    }

    @Test
    public void testQuantileEstimator() {
        int nTrials = 10_000;
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 100, 50, 0.26);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 500, 50, 0.18);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 1_000, 50, 0.15);

        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 100, 95, 0.08);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 500, 95, 0.05);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 1_000, 95, 0.04);

        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 100, 99, 0.05);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 500, 99, 0.03);
        testQuantileEstimator(RandomStreamCreator.UNIFORM, nTrials, 1_000, 99, 0.02);

        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 100, 50, 0.20);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 500, 50, 0.14);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 1_000, 50, 0.12);

        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 100, 95, 0.07);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 500, 95, 0.04);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 1_000, 95, 0.04);

        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 100, 99, 0.18);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 500, 99, 0.06);
        testQuantileEstimator(RandomStreamCreator.GAUSSIAN, nTrials, 1_000, 99, 0.04);
    }
}
