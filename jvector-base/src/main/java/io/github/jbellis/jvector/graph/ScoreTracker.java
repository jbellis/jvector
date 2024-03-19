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

import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import static java.lang.Math.max;

interface ScoreTracker {
    VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

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
        static final int RECENT_SCORES_TRACKED = 100;

        // a sliding window of recent scores
        private final VectorFloat<?> recentScores;
        private int recentEntryIndex;

        // observation count
        private int observationCount;
        private float bestMean;

        private final float threshold;

        TwoPhaseTracker(float threshold) {
            this.recentScores = vts.createFloatVector(RECENT_SCORES_TRACKED);
            this.threshold = threshold;
        }

        @Override
        public void track(float score) {
            recentScores.set(recentEntryIndex, score);
            recentEntryIndex = (recentEntryIndex + 1) % recentScores.length();
            observationCount++;
        }

        @Override
        public boolean shouldStop() {
            // don't stop if we don't have enough data points
            if (observationCount < RECENT_SCORES_TRACKED) {
                return false;
            }
            // evaluation is expensive so only do it 10% of the time
            if (observationCount % 10 != 0) {
                return false;
            }

            // we're in phase 2 if the median of the recent scores is worse than the worst best score,
            // indicating that we found the local maximum and are unlikely to find better options
            float windowMean = VectorUtil.sum(recentScores) / recentScores.length();
            try {
                return windowMean < bestMean && windowMean < threshold;
            } finally {
                bestMean = max(bestMean, windowMean);
            }
        }
    }
}
