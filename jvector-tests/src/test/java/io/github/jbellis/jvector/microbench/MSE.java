package io.github.jbellis.jvector.microbench;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.LVQ;
import io.github.jbellis.jvector.pq.LocallyAdaptiveVectorQuantization;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MSE {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void main(String[] args) throws IOException {
        // compute for the cohere-100k dataset
        var mfd = DownloadHelper.maybeDownloadFvecs("cohere-english-v3-100k");
        var dataset = mfd.load();

        // three stage PQ
        var pq8 = ProductQuantization.compute(dataset.getBaseRavv(), dataset.getDimension() / 2, 256, false);
        var pq8Encoded = pq8.encodeAll(dataset.getBaseRavv());
        List<VectorFloat<?>> residuals = IntStream.range(0, dataset.baseVectors.size()).parallel().mapToObj(i -> {
            var decoded = vts.createFloatVector(dataset.getDimension());
            pq8.decode(pq8Encoded[i], decoded);
            return VectorUtil.sub(dataset.baseVectors.get(i), decoded);
        }).collect(Collectors.toList());
        var residualVV = new ListRandomAccessVectorValues(residuals, dataset.getDimension());
        var pqR = ProductQuantization.compute(residualVV, dataset.getDimension() / 8, 256, false);
        var pqREncoded = pqR.encodeAll(residualVV);
        List<VectorFloat<?>> residuals2 = IntStream.range(0, dataset.baseVectors.size()).parallel().mapToObj(i -> {
            var decoded = vts.createFloatVector(dataset.getDimension());
            pq8.decode(pq8Encoded[i], decoded);
            var residual = vts.createFloatVector(dataset.getDimension());
            pqR.decode(pqREncoded[i], residual);
            VectorUtil.addInPlace(decoded, residual);
            return VectorUtil.sub(dataset.baseVectors.get(i), decoded);
        }).collect(Collectors.toList());
        var residualVV2 = new ListRandomAccessVectorValues(residuals2, dataset.getDimension());
        var pqR2 = ProductQuantization.compute(residualVV2, dataset.getDimension() / 32, 256, false);
        var pqREncoded2 = pqR2.encodeAll(residualVV2);

        // compute MSE
        double pqError = IntStream.range(0, dataset.baseVectors.size()).parallel().mapToDouble(i -> {
            var decoded = vts.createFloatVector(dataset.getDimension());
            pq8.decode(pq8Encoded[i], decoded);
            var residual = vts.createFloatVector(dataset.getDimension());
            pqR.decode(pqREncoded[i], residual);
            VectorUtil.addInPlace(decoded, residual);
            pqR2.decode(pqREncoded2[i], residual);
            VectorUtil.addInPlace(decoded, residual);
            return VectorUtil.squareL2Distance(dataset.baseVectors.get(i), decoded);
        }).sum();

        System.out.println("PQ error: " + pqError);
    }
}
