package com.github.jbellis.jvector.microbench;


import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import com.github.jbellis.jvector.vector.SimdOps;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1)
public class SimilarityBench {

    static int SIZE = 256;
    static final float[] q1 = new float[SIZE];
    static final float[] q2 = new float[SIZE];

    static final float[] q3 = new float[2];

    static {
        for (int i = 0; i < q1.length; i++) {
            q1[i] = ThreadLocalRandom.current().nextFloat();
            q2[i] = ThreadLocalRandom.current().nextFloat();
        }

        q3[0] = ThreadLocalRandom.current().nextFloat();
        q3[1] = ThreadLocalRandom.current().nextFloat();
    }

    @State(Scope.Benchmark)
    public static class Parameters {

    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProduct(Blackhole bh, Parameters p) {
        bh.consume(SimdOps.dotProduct(q3, 0, q1, 22, q3.length));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
