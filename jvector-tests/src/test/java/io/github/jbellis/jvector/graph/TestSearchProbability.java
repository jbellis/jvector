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
            testFutureProbability(random::nextGaussian, 0.5f, GraphSearcher.RECENT_SCORES_TRACKED);

            // normal dist w/ offset + scale
            int offset = random.nextInt(10);
            int scale = nextInt(5, 10);
            testFutureProbability(() -> offset + scale * random.nextGaussian(), random.nextDouble(), GraphSearcher.RECENT_SCORES_TRACKED);
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
        double predictedProbability = GraphSearcher.futureProbabilityAboveThreshold(recentSimilarities, threshold);
        assertEquals(observedProbability, predictedProbability, 0.1);  // Allowing 10% error
    }
}
