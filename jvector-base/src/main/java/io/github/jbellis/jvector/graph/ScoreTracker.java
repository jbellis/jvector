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

import io.github.jbellis.jvector.util.AbstractLongHeap;
import io.github.jbellis.jvector.util.BoundedLongHeap;
import org.apache.commons.math3.stat.StatUtils;

import static io.github.jbellis.jvector.util.NumericUtils.floatToSortableInt;
import static io.github.jbellis.jvector.util.NumericUtils.sortableIntToFloat;

interface ScoreTracker {

    ScoreTracker NO_OP = new NoOpTracker();

    void track(float score);

    boolean shouldStop();

    class NoOpTracker implements ScoreTracker {
        @Override
        public void track(float score) { }

        @Override
        public boolean shouldStop() {
            return false;
        }
    }

    /**
     * Follows the methodology of section 3.1 in "VBase: Unifying Online Vector Similarity Search
     * and Relational Queries via Relaxed Monotonicity" to determine when we've left phase 1
     * (finding the local maximum) and entered phase 2 (mostly just finding worse options)
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
        public boolean shouldStop() {
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
            double worstBest = sortableIntToFloat((int) bestScores.top());
            return windowMedian < worstBest && windowMedian < threshold;
        }
    }
}
