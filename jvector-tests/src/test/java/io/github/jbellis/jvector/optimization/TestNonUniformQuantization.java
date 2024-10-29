package io.github.jbellis.jvector.optimization;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.Test;

import java.util.Random;
import java.util.stream.DoubleStream;

import io.github.jbellis.jvector.vector.VectorUtil;
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
        float constant = (float) (Math.pow(2, nBits) - 1);
        VectorUtil.scale(x, constant);
        for (int d = 0; d < x.length(); d++) {
            x.set(d, (float) Math.round(x.get(d)));
        }
    }

    // In-place dequantization
    private void uniformDequantize(VectorFloat<?> x, int nBits) {
        float constant = (float) (Math.pow(2, nBits) - 1);
        VectorUtil.scale(x, 1.f / constant);
    }

    // In-place application of the CDF of the Kumaraswamy distribution
    private void forwardKumaraswamy(VectorFloat<?> x, float a, float b) {
        // Compute 1 - (1 - v ** a) ** b
        VectorUtil.constantMinusExponentiatedVector(x, 1, a); // 1 - v ** a
        VectorUtil.constantMinusExponentiatedVector(x, 1, b); // 1 - v ** b
    }

    // In-place application of the inverse CDF of the Kumaraswamy distribution
    private void inverseKumaraswamy(VectorFloat<?> y, float a, float b) {
        // Compute (1 - (1 - y) ** (1 / b)) ** (1 / a)
        VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / b); // 1 - v ** (1 / a)
        VectorUtil.exponentiateConstantMinusVector(y, 1, 1.f / a); // 1 - v ** (1 / b)
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
            forwardKumaraswamy(vectorCopy, (float) x[0], (float) x[1]);
            uniformQuantize(vectorCopy, nBits);
            uniformDequantize(vectorCopy, nBits);
            inverseKumaraswamy(vectorCopy, (float) x[0], (float) x[1]);

            return -VectorUtil.squareL2Distance(vectorOriginal, vectorCopy);
        }
    }

    public double uniformQuantizationLoss(VectorFloat<?> vector, int nBits) {
        var vectorCopy = vectorTypeSupport.createFloatVector(vector.length());
        vectorCopy.copyFrom(vector, 0, 0, vector.length());
        uniformQuantize(vectorCopy, nBits);
        uniformDequantize(vectorCopy, nBits);

        return VectorUtil.squareL2Distance(vector, vectorCopy);
    }

    @Test
    public void testGaussian() {
        {
            var nDims = 3096;
            var nBits = 8;
            var nTrials = 50;

            var uniformError = new double[nTrials];
            var kumaraswamyError = new double[nTrials];

            for (int trial = 0; trial < nTrials; trial++) {
                var vector = gaussianVector(getRandom(), nDims);
                var min = VectorUtil.min(vector);
                var max = VectorUtil.max(vector);
                VectorUtil.subInPlace(vector, min);
                VectorUtil.scale(vector, 1.f / (max - min));

                var loss = new KumaraswamyQuantizationLossFunction(2, nBits, vector);
                loss.setMinBounds(new double[]{1e-6, 1e-6});

                double[] initialSolution = {1, 1};
                var xnes = new NESOptimizer(NESOptimizer.Distribution.SEPARABLE);

                var tolerance = 1e-6;
                xnes.setTol(tolerance);
                var sol = xnes.optimize(loss, initialSolution, 0.5);

                uniformError[trial] = uniformQuantizationLoss(vector, nBits);
                kumaraswamyError[trial] = -1 * loss.compute(sol.x);
            }

            var averageUniformError = DoubleStream.of(uniformError).average().getAsDouble();
            var averageKumaraswamyError = DoubleStream.of(kumaraswamyError).average().getAsDouble();

            var stdUniformError = Math.sqrt(DoubleStream.of(uniformError).map(e -> Math.pow(e - averageUniformError, 2)).average().getAsDouble());
            var stdKumaraswamyError = Math.sqrt(DoubleStream.of(kumaraswamyError).map(e -> Math.pow(e - averageKumaraswamyError, 2)).average().getAsDouble());

            System.out.println("Uniform reconstruction loss=" + averageUniformError + "  STD=" + stdUniformError);
            System.out.println("Kumaraswamy reconstruction loss=" + averageKumaraswamyError + "  STD=" + stdKumaraswamyError);

            System.out.println("Ratio=" + (averageUniformError / averageKumaraswamyError));

            // This tolerance is too conservative, the value should in reality be >1.7 with a high number of samples.
            // Keeping it this way for speed
            var testTolerance = 1.6;
            assertTrue(
                    "Uniform AVG reconstruction error=" + averageUniformError +
                            " Kumaraswamy reconstruction loss=" + averageKumaraswamyError,
                    (averageUniformError / averageKumaraswamyError) >= testTolerance);
        }
    }
}
