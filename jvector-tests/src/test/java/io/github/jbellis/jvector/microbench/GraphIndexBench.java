package io.github.jbellis.jvector.microbench;


import java.util.Random;
import java.util.concurrent.TimeUnit;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = "--add-modules=jdk.incubator.vector")
public class GraphIndexBench {

    static float[][] createRandomFloatVectors(int size, int dimension, Random random) {
        float[][] vectors = new float[size][];
        for (int offset = 0; offset < size; offset++) {
            vectors[offset] = TestUtil.randomVector(random, dimension);
        }
        return vectors;
    }

    static class TestRandomAccessReader implements RandomAccessVectorValues<float[]> {
        private final float[][] values;

        TestRandomAccessReader(float[][] values) {
            this.values = values;
        }

        @Override
        public int size() {
            return values.length;
        }

        @Override
        public int dimension() {
            return values[0].length;
        }

        @Override
        public float[] vectorValue(int targetOrd) {
            return values[targetOrd];
        }

        @Override
        public boolean isValueShared() {
            return false;
        }

        @Override
        public RandomAccessVectorValues<float[]> copy() {
            return new TestRandomAccessReader(values);
        }
    }

    @State(Scope.Benchmark)
    public static class Parameters {
        final Random r;
        final TestRandomAccessReader vectors;

        public Parameters() {
            this.r = new Random(1337);
            this.vectors = new TestRandomAccessReader(createRandomFloatVectors(100000, 1536, r));
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testGraphBuild(Blackhole bh, Parameters p) {
        GraphIndexBuilder<float[]> graphIndexBuilder =  new GraphIndexBuilder<>(p.vectors, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, 8, 60, 1.2f, 1.4f);
        bh.consume(graphIndexBuilder.build());
    }
}

