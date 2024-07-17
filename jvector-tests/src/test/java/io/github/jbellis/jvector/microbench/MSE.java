package io.github.jbellis.jvector.microbench;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.example.util.DownloadHelper;
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

        var pq = ProductQuantization.compute(dataset.getBaseRavv(), (dataset.getDimension() * 2) / 3, 256, false);
        var pqEncoded = pq.encodeAll(dataset.getBaseRavv());

        // encode LVQ
        var lvq = LocallyAdaptiveVectorQuantization.compute(dataset.getBaseRavv());
        var lvqEncoded = lvq.encodeAll(dataset.getBaseRavv());

        // compute MSE vs uncompressed for both
        double pqError = IntStream.range(0, dataset.baseVectors.size()).parallel().mapToDouble(i -> {
            var decoded = vts.createFloatVector(dataset.getDimension());
            pq.decode(pqEncoded[i], decoded);
            return VectorUtil.squareL2Distance(dataset.baseVectors.get(i), decoded);
        }).sum();
        double lvqError = IntStream.range(0, dataset.baseVectors.size()).parallel().mapToDouble(i -> {
            var v = dataset.baseVectors.get(i).copy();
            VectorUtil.subInPlace(v, lvq.globalMean);
            return VectorUtil.squareL2Distance(v, lvqEncoded[i].decode());
        }).sum();

        System.out.println("PQ error: " + pqError);
        System.out.println("LVQ error: " + lvqError);
    }
}
