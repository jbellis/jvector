package io.github.jbellis.jvector.microbench;

import java.util.concurrent.TimeUnit;

import io.github.jbellis.jvector.util.PoolingSupport;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1)
public class PoolingBench {

    private static final PoolingSupport<float[]> queue = PoolingSupport.newQueuePooling(32, () -> new float[1024]);
    private static final PoolingSupport<float[]> pool = PoolingSupport.newThreadBased(() -> new float[1024]);
    private static final ThreadLocal<float[]> tlocal = ThreadLocal.withInitial(() -> new float[1024]);
    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void noPooling(Blackhole bh) {
        bh.consume(new float[1024]);
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void localpooling(Blackhole bh) {
        try(var pooled = pool.get()) {
            bh.consume(pooled.get());
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void queuepooling(Blackhole bh) {
        try(var pooled = queue.get()) {
            bh.consume(pooled.get());
        }
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void threadLocals(Blackhole bh) {
        var a  = tlocal.get();
        bh.consume(a);
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
