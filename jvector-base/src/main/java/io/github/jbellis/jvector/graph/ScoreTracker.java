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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;

interface ScoreTracker {

    ScoreTracker NO_OP = new NoOpTracker();

    void track(float score);

    boolean shouldStop(int numVisited);

    class NoOpTracker implements ScoreTracker {
        @Override
        public void track(float score) { }

        @Override
        public boolean shouldStop(int numVisited) {
            return false;
        }
    }

    class NormalDistributionTracker implements ScoreTracker {
        @VisibleForTesting
        // in TestSearchProbability, 100 is not enough to stay within a 10% error rate, but 300 is
        static final int RECENT_SCORES_TRACKED = 300;

        private final double[] recentScores;
        private int index;
        private final double threshold;

        NormalDistributionTracker(double threshold) {
            this.recentScores = new double[RECENT_SCORES_TRACKED];
            this.threshold = threshold;
        }

        @Override
        public void track(float score) {
            recentScores[index] = score;
            index = (index + 1) % recentScores.length;
        }

        @Override
        public boolean shouldStop(int numVisited) {
            if (numVisited < recentScores.length) {
                return false;
            }
            return numVisited % 100 == 0 && futureProbabilityAboveThreshold(recentScores, threshold) < 0.01;
        }

        /**
         * Return the probability of finding a node above the given threshold in the future,
         * given the similarities observed recently.
         */
        @VisibleForTesting
        static double futureProbabilityAboveThreshold(double[] recentSimilarities, double threshold) {
            // Calculate sample mean and standard deviation
            double sampleMean = StatUtils.mean(recentSimilarities);
            double sampleStd = Math.sqrt(StatUtils.variance(recentSimilarities));

            // Z-score for the threshold
            double zScore = (threshold - sampleMean) / sampleStd;

            // Probability of finding a node above the threshold in the future
            NormalDistribution normalDistribution = new NormalDistribution(sampleMean, sampleStd);
            return 1 - normalDistribution.cumulativeProbability(zScore);
        }
    }
}
