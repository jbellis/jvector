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

import io.github.jbellis.jvector.util.BoundedLongHeap;
import org.apache.commons.math3.stat.StatUtils;

import static io.github.jbellis.jvector.util.NumericUtils.floatToSortableInt;
import static io.github.jbellis.jvector.util.NumericUtils.sortableIntToFloat;

interface ScoreTracker {

    ScoreTracker NO_OP = new NoOpTracker();

    void track(float score);

    boolean shouldStop(double worstBest);

    class NoOpTracker implements ScoreTracker {
        @Override
        public void track(float score) { }

        @Override
        public boolean shouldStop(double worstBestScore) {
            return false;
        }
    }

    /**
     * Follows the methodology of section 3.1 in "VBase: Unifying Online Vector Similarity Search
     * and Relational Queries via Relaxed Monotonicity" to determine when we've left phase 1
     * (finding the local maximum) and entered phase 2 (mostly just finding worse options)
     * This implementation does not consider the worstBestScore provided to shouldStop.
     */
    class TwoPhaseTracker implements ScoreTracker {
        static final int RECENT_SCORES_TRACKED = 500;
        static final int BEST_SCORES_TRACKED = 100;

        // a sliding window of recent scores
        private final double[] recentScores;
        private int recentEntryIndex;

        // Heap of the best scores seen so far
        BoundedLongHeap bestScores;

        // observation count
        private int observationCount;

        private final double threshold;

        TwoPhaseTracker(double threshold) {
            this.recentScores = new double[RECENT_SCORES_TRACKED];
            this.bestScores = new BoundedLongHeap(BEST_SCORES_TRACKED);
            this.threshold = threshold;
        }

        @Override
        public void track(float score) {
            bestScores.push(floatToSortableInt(score));
            recentScores[recentEntryIndex] = score;
            recentEntryIndex = (recentEntryIndex + 1) % recentScores.length;
            observationCount++;
        }

        @Override
        public boolean shouldStop(double worstBestScore) {
            // don't stop if we don't have enough data points
            if (observationCount < RECENT_SCORES_TRACKED) {
                return false;
            }
            // evaluation is expensive so only do it 1% of the time
            if (observationCount % 100 != 0) {
                return false;
            }

            // We're in phase 2 if the 99th percentile of the recent scores evaluated is lower
            // than the worst of the best scores seen.
            //
            // (paper suggests using the median of recent scores, but experimentally that is too prone to false positives.
            // 90th does seem to be enough, but 99th doesn't result in much extra work, so we'll be conservative)
            double windowMedian = StatUtils.percentile(recentScores, 99);
            worstBestScore = sortableIntToFloat((int) bestScores.top());
            return windowMedian < worstBestScore && windowMedian < threshold;
        }
    }

    /**
     * Follows the methodology of section 3.1 in "VBase: Unifying Online Vector Similarity Search
     * and Relational Queries via Relaxed Monotonicity" to determine when we've left phase 1
     * (finding the local maximum) and entered phase 2 (mostly just finding worse options)
     * To compute quantiles quickly, we treat the distribution of the data as Normal,
     * track its mean and variance, and compute quantiles from them as:
     *     mean + SIGMA_FACTOR * sqrt(variance)
     * Empirically, SIGMA_FACTOR=4 seems to work reasonably well.
     */
    class RelaxedMonotonicityTracker implements ScoreTracker {
        static final double SIGMA_FACTOR = 4;

        // a sliding window of recent scores
        private final double[] recentScores;
        private int recentEntryIndex;

        // observation count
        private int observationCount;

        // the sample mean
        private double mean;

        // the sample variance multiplied by n-1
        private double dSquared;

        /**
         * Constructor
         * @param recentScoredTracked the number of tracked scores used to estimate if we are unlikely to improve
         *                            the results anymore. An empirical rule of thumb is 3 * rerankK.
         */
        RelaxedMonotonicityTracker(int recentScoredTracked) {
            this.recentScores = new double[recentScoredTracked];
            this.mean = 0;
            this.dSquared = 0;
        }

        @Override
        public void track(float score) {
            observationCount++;

            // The updates follow the math in
            // https://nestedsoftware.com/2019/09/26/incremental-average-and-standard-deviation-with-sliding-window-470k.176143.html
            if (observationCount <= this.recentScores.length) {
                // if the buffer is not full yet, use standard Welford method
                var meanDelta = (score - this.mean) / (observationCount);
                var newMean = this.mean + meanDelta;

                var dSquaredDelta = ((score - newMean) * (score - this.mean));
                var newDSquared = this.dSquared + dSquaredDelta;

                this.mean = newMean;
                this.dSquared = newDSquared;
            } else {
                // once the buffer is full, adjust Welford Method for window size
                var oldScore = recentScores[recentEntryIndex];
                var meanDelta = (score - oldScore) / this.recentScores.length;
                var newMean = this.mean + meanDelta;

                var dSquaredDelta = ((score - oldScore) * (score - newMean + oldScore - this.mean));
                var newDSquared = this.dSquared + dSquaredDelta;

                this.mean = newMean;
                this.dSquared = newDSquared;
            }
            recentScores[recentEntryIndex] = score;
            recentEntryIndex = (recentEntryIndex + 1) % recentScores.length;

        }

        @Override
        public boolean shouldStop(double worstBestScore) {
            // don't stop if we don't have enough data points
            if (observationCount < this.recentScores.length) {
                return false;
            }

            // We're in phase 2 if the q-th percentile of the recent scores evaluated,
            //     mean + SIGMA_FACTOR * sqrt(variance)
            // is lower than the worst of the best scores seen.
            // (paper suggests using the median of recent scores, but experimentally that is too prone to false positives)
            double std = Math.sqrt(this.dSquared / (this.recentScores.length - 1));
            double windowPercentile = this.mean + SIGMA_FACTOR * std;

            return windowPercentile < worstBestScore;
        }
    }
}
