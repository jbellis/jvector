package io.github.jbellis.jvector.optimization;

import io.github.jbellis.jvector.util.MathUtil;
import org.junit.Test;

import java.util.stream.DoubleStream;

import static org.junit.Assert.assertTrue;

public class TestNaturalEvolutionStrategies {
    @Test
    public void testQuadraticOptimization() {
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
        var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

        var tolerance = 1e-9;
        xnes.setTol(tolerance);
        var sol = xnes.optimize(loss, initialSolution, 0.5);

        assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
        assertTrue("sol.x[0]=" + sol.x[0], sol.x[0] < 1e-4);
        assertTrue("sol.x[1]=" + sol.x[1], sol.x[1] < 1e-4);


    }
}
