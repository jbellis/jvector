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
        testFutureProbability(random::nextGaussian, 0.5f, 100);
        int offset = random.nextInt(10);
        int scale = nextInt(5, 10);
        testFutureProbability(() -> offset + scale * random.nextGaussian(), random.nextDouble(), 100);
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
