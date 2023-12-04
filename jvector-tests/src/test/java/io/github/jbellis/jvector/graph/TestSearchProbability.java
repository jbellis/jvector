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

import io.github.jbellis.jvector.LuceneTestCase;
import org.junit.Test;

import java.util.Random;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestSearchProbability extends LuceneTestCase {

    @Test
    public void testFutureProbabilityAboveThreshold() {
        Random random = getRandom();

        for (int i = 0; i < 10; i++) {
            // standard normal distribution
            testFutureProbability(random::nextGaussian, 0.5f, ScoreTracker.NormalDistributionTracker.RECENT_SCORES_TRACKED);

            // normal dist w/ offset + scale
            int offset = random.nextInt(10);
            int scale = nextInt(5, 10);
            testFutureProbability(() -> offset + scale * random.nextGaussian(), random.nextDouble(), ScoreTracker.NormalDistributionTracker.RECENT_SCORES_TRACKED);
        }
    }

    private void testFutureProbability(Supplier<Double> generator, double threshold, int numSamples) {
        // generate similarites and count the number above the threshold
        double[] recentSimilarities = new double[numSamples];
        int n = 0;
        for (int i = 0; i < numSamples; i++) {
            recentSimilarities[i] = generator.get();
            if (recentSimilarities[i] > threshold) {
                n++;
            }
        }
        double observedProbability = (double) n / numSamples;

        // check the prediction vs the observed
        double predictedProbability = ScoreTracker.NormalDistributionTracker.futureProbabilityAboveThreshold(recentSimilarities, threshold);
        assertEquals(observedProbability, predictedProbability, 0.11);  // Allowing 11% error b/c 10% occasionally fails
    }
}
