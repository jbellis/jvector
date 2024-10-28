package io.github.jbellis.jvector.optimization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;

import static io.github.jbellis.jvector.util.MathUtil.square;
import static io.github.jbellis.jvector.vector.VectorUtil.min;
import static io.github.jbellis.jvector.vector.VectorUtil.max;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestNonUniformQuantization extends RandomizedTest {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    private static VectorFloat<?> gaussianVector(Random random, int dim) {
        var vec = vectorTypeSupport.createFloatVector(dim);
        for (int i = 0; i < dim; i++) {
            vec.set(i, (float) random.nextGaussian());
        }
        return vec;
    }

    // In-place quantization
    private void uniformQuantize(VectorFloat<?> x, int nBits) {
        double constant = Math.pow(2, nBits) - 1;
        for (int d = 0; d < x.length(); d++) {
            x.set(d, (float) Math.round(constant * x.get(d)));
        }
    }

    // In-place dequantization
    private void uniformDequantize(VectorFloat<?> x, int nBits) {
        double constant = Math.pow(2, nBits) - 1;
        for (int d = 0; d < x.length(); d++) {
            x.set(d, (float) (x.get(d) / constant));
        }
    }

    // In-place application of the CDF of the Kumaraswamy distribution
    private void forwardKumaraswamy(VectorFloat<?> x, double a, double b) {
        // Compute 1 - (1 - x ** a) ** b
        for (int d = 0; d < x.length(); d++) {
            x.set(d, (float) (1.f - Math.pow(1 - Math.pow(x.get(d), a), b)));
        }
    }

    // In-place application of the inverse CDF of the Kumaraswamy distribution
    private void inverseKumaraswamy(VectorFloat<?> y, double a, double b) {
        // Compute (1 - (1 - y) ** (1 / b)) ** (1 / a)
        for (int d = 0; d < y.length(); d++) {
            y.set(d, (float) Math.pow(1 - Math.pow(1 - y.get(d), 1. / b), 1. / a));
        }
    }

    class KumaraswamyQuantizationLossFunction extends LossFunction {
        final private int nBits;
        final private VectorFloat<?> vectorOriginal;
        final private VectorFloat<?> vectorCopy;

        public KumaraswamyQuantizationLossFunction(int nDims, int nBits, VectorFloat<?> vector) {
            super(nDims);
            this.nBits = nBits;
            vectorOriginal = vector;
            vectorCopy = vectorTypeSupport.createFloatVector(vectorOriginal.length());
        }

        public double compute(double[] x) {
            vectorCopy.copyFrom(vectorOriginal, 0, 0, vectorOriginal.length());
            forwardKumaraswamy(vectorCopy, x[0], x[1]);
            uniformQuantize(vectorCopy, nBits);
            uniformDequantize(vectorCopy, nBits);
            inverseKumaraswamy(vectorCopy, x[0], x[1]);

            double lossValue = 0;
            for (int d = 0; d < vectorOriginal.length(); d++) {
                lossValue -= square(vectorOriginal.get(d) - vectorCopy.get(d));
            }
            return lossValue;
        }
    }

    public double uniformQuantizationLoss(VectorFloat<?> vector, int nBits) {
        var vectorCopy = vectorTypeSupport.createFloatVector(vector.length());
        vectorCopy.copyFrom(vector, 0, 0, vector.length());
        uniformQuantize(vectorCopy, nBits);
        uniformDequantize(vectorCopy, nBits);

        double lossValue = 0;
        for (int d = 0; d < vector.length(); d++) {
            lossValue -= square(vector.get(d) - vectorCopy.get(d));
        }
        return lossValue;
    }

    @Test
    public void testGaussian() {
        {
            var nBits = 8;

            var vector = gaussianVector(getRandom(), 3096);
            var min = min(vector);
            var max = max(vector);
            for (int d = 0; d < vector.length(); d++) {
                vector.set(d, (vector.get(d) - min) / (max - min));
            }

            var loss = new KumaraswamyQuantizationLossFunction(2, nBits, vector);
            loss.setMinBounds(new double[]{1e-6, 1e-6});

            double[] initialSolution = {1, 1};
            var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);
//            xnes.setNSamples(50);

            var tolerance = 1e-6;
            xnes.setTol(tolerance);
            var sol = xnes.optimize(loss, initialSolution, 0.5);

            System.out.println("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1]);
            System.out.println("error=" + sol.error);
            System.out.println("Kumaraswamy reconstruction loss=" + loss.compute(sol.x));

            System.out.println("uniform reconstruction loss=" + uniformQuantizationLoss(vector, nBits));

//            assertTrue("error=" + sol.error + " tolerance=" + tolerance, sol.error <= tolerance);
//            assertTrue("sol.x[0]=" + sol.x[0] + " sol.x[1]=" + sol.x[1], sol.x[0] < 1e-3 && sol.x[1] < 1e-3);
        }
    }
}
