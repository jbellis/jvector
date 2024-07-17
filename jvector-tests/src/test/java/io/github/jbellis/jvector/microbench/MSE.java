package io.github.jbellis.jvector.microbench;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleReader;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MSE {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static List<VectorFloat<?>> createRandomVectors(int count, int dimension) {
        return IntStream.range(0, count).mapToObj(i -> TestUtil.randomVector(ThreadLocalRandom.current(), dimension)).collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException {
        // compute for the cohere-100k dataset
        var mfd = DownloadHelper.maybeDownloadFvecs("cohere-english-v3-100k");
        var dataset = mfd.load();

        // PQ is precomputed
        var pqv = PQVectors.load(new SimpleReader(Path.of("/home/jonathan/Projects/cuvs/cohere.pqv")));

        // encode LVQ
        var lvq = new

        // sanity check
        var vsf = VectorSimilarityFunction.DOT_PRODUCT;
        var zeros = new float[1024];
        var zeroScores = pqv.scoreFunctionFor(vts.createFloatVector(zeros), vsf);
        var ones = new float[1024];
        Arrays.fill(ones, 1.0f);
        var oneScores = pqv.scoreFunctionFor(vts.createFloatVector(ones), vsf);
        System.out.println("Zeroes:");
        for (int i = 0; i < 10; i++) {
            System.out.println(zeroScores.similarityTo(i));
        }
        System.out.println("Ones:");
        for (int i = 0; i < 10; i++) {
            System.out.println(oneScores.similarityTo(i));
        }

        int dim = pqv.getOriginalSize() / Float.BYTES;
        for (int i = 0; i < 10_000; i++) {
            VectorFloat<?> q = TestUtil.randomVector(ThreadLocalRandom.current(), dim);
            runSimple(pqv, q);
            runPrecomputed(pqv, q);
        }
        System.out.println("Warmup complete");

        long timeSimple = 0;
        long timePrecomputed = 0;
        for (int i = 0; i < 10_000; i++) {
            VectorFloat<?> q = TestUtil.randomVector(ThreadLocalRandom.current(), dim);
            timeSimple += runSimple(pqv, q);
            timePrecomputed += runPrecomputed(pqv, q);
        }
        System.out.println("Simple: " + timeSimple / 1_000_000 + "ms");
        System.out.println("Precomputed: " + timePrecomputed / 1_000_000 + "ms");
    }

    private static long runSimple(PQVectors pqv, VectorFloat<?> q) {
        var R = ThreadLocalRandom.current();
        long start;
        start = System.nanoTime();
        var sf = pqv.scoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);
        for (int j = 0; j < 32 * 50; j++) {
            sf.similarityTo(R.nextInt(pqv.count()));
        }
        return System.nanoTime() - start;
    }

    private static long runPrecomputed(PQVectors pqv, VectorFloat<?> q) {
        var R = ThreadLocalRandom.current();
        long start;
        start = System.nanoTime();
        var sf2 = pqv.precomputedScoreFunctionFor(q, VectorSimilarityFunction.DOT_PRODUCT);
        for (int j = 0; j < 32 * 50; j++) {
            sf2.similarityTo(R.nextInt(pqv.count()));
        }
        return System.nanoTime() - start;
    }
}
