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
        assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], sol.x[0] < 1e-3 && sol.x[1] < 1e-3);
    }

    @Test
    public void testQuarticOptimization() {
        class TestLossFunction extends LossFunction {
            public TestLossFunction(int nDims) {
                super(nDims);
            }

            public double compute(double[] x) {
                return -1 * DoubleStream.of(x).map(elem -> Math.pow(elem, 4) - Math.pow(elem, 2) - 0.25 * elem).sum();
            }
        }
        var loss = new TestLossFunction(2);
        loss.setMinBounds(new double[] {-1000, -1000});
        loss.setMaxBounds(new double[] {1000, 1000});


        var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

        var tolerance = 1e-9;
        xnes.setTol(tolerance);

        {
            // Setting the initial solution close to the global minimum works
            var sol = xnes.optimize(loss, new double[] {1, 1});

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
        }

        {
            // Setting the initial solution close to the global minimum works
            var sol = xnes.optimize(loss, new double[] {1, 1}, 0.5);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
        }

        {
            // Setting the initial solution close to a local minimum makes it fail
            var sol = xnes.optimize(loss, new double[]{-1, -1}, 0.2);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) >= 1e-3 || Math.abs(sol.x[1] - 0.76284314) >= 1e-3);
        }

        {
            // Increasing the variance of for the natural gradient makes it work again
            var sol = xnes.optimize(loss, new double[] {-1, -1}, 1);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
            }

    }
}
