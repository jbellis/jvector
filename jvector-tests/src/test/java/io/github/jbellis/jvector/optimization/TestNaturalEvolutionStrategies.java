package io.github.jbellis.jvector.optimization;

import io.github.jbellis.jvector.util.MathUtil;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class TestNaturalEvolutionStrategies {
    @Test
    public void testQuadraticOptimization() {
        class TestLossFunction extends LossFunction {
            public TestLossFunction(int nDims) {
                super(nDims);
            }

            public float compute(float[] x) {
                float sum = 0;
                for (float num : x) {
                    sum -= MathUtil.square(num);
                }
                return sum;
            }
        }

        {
            // Box constraints are specified
            var loss = new TestLossFunction(2);
            loss.setMinBounds(new float[]{-1000, -1000});
            loss.setMaxBounds(new float[]{1000, 1000});

            float[] initialSolution = {1, 1};
            var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

            var tolerance = 1e-9f;
            xnes.setTol(tolerance);
            var sol = xnes.optimize(loss, initialSolution, 0.5f);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], sol.x[0] < 1e-3 && sol.x[1] < 1e-3);
        }

        {
            // No box constraints are specified
            var loss = new TestLossFunction(2);

            float[] initialSolution = {1, 1};
            var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

            var tolerance = 1e-9f;
            xnes.setTol(tolerance);
            var sol = xnes.optimize(loss, initialSolution, 0.5f);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], sol.x[0] < 1e-3 && sol.x[1] < 1e-3);
        }
    }

    @Test
    public void testQuarticOptimization() {
        class TestLossFunction extends LossFunction {
            public TestLossFunction(int nDims) {
                super(nDims);
            }

            public float compute(float[] x) {
                float sum = 0;
                for (float elem : x) {
                    sum -= (float) (Math.pow(elem, 4) - Math.pow(elem, 2) - 0.25 * elem);
                }
                return sum;
            }
        }
        var loss = new TestLossFunction(2);
        loss.setMinBounds(new float[] {-1000, -1000});
        loss.setMaxBounds(new float[] {1000, 1000});


        var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

        var tolerance = 1e-9f;
        xnes.setTol(tolerance);

        {
            // Setting the initial solution close to the global minimum works
            var sol = xnes.optimize(loss, new float[] {1, 1});

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
        }

        {
            // Setting the initial solution close to the global minimum works
            var sol = xnes.optimize(loss, new float[] {1, 1}, 0.5f);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
        }

        {
            // Setting the initial solution close to a local minimum makes it fail
            var sol = xnes.optimize(loss, new float[]{-1, -1}, 0.2f);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) >= 1e-3 || Math.abs(sol.x[1] - 0.76284314) >= 1e-3);
        }

        {
            // Increasing the variance of for the natural gradient makes it work again
            var sol = xnes.optimize(loss, new float[] {-1, -1}, 1);

            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], Math.abs(sol.x[0] - 0.76284314) < 1e-3 && Math.abs(sol.x[1] - 0.76284314) < 1e-3);
            }

    }
}
