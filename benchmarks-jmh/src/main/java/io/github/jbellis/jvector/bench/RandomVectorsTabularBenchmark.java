package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.bench.output.TableRepresentation;
import io.github.jbellis.jvector.bench.output.TextTable;
import io.github.jbellis.jvector.example.SiftSmall;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class RandomVectorsTabularBenchmark {
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();
    private RandomAccessVectorValues ravv;

    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> queryVectors;
    private GraphIndexBuilder graphIndexBuilder;
    private GraphIndex graphIndex;
    private int originalDimension;
    private long totalTransactions;

    private final AtomicLong transactionCount = new AtomicLong(0);
    private final AtomicLong totalLatency = new AtomicLong(0);
    private final Queue<Long> latencySamples = new ConcurrentLinkedQueue<>(); // Store latencies for P99.9
    private final Queue<Integer> visitedSamples = new ConcurrentLinkedQueue<>();
    private final Queue<Long> recallSamples = new ConcurrentLinkedQueue<>();
    private ScheduledExecutorService scheduler;
    private long startTime;

    private final TableRepresentation tableRepresentation = new TextTable(); // Keep TextTable only for now

    @Param({"1000", "10000", "100000", "1000000"})
    int numBaseVectors;

    @Param({"10"})
    int numQueryVectors;

    @Setup
    public void setup() throws IOException {
        originalDimension = 128;

        baseVectors = new ArrayList<>(numBaseVectors);
        queryVectors = new ArrayList<>(numQueryVectors);

        for (int i = 0; i < numBaseVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            baseVectors.add(vector);
        }

        for (int i = 0; i < numQueryVectors; i++) {
            VectorFloat<?> vector = createRandomVector(originalDimension);
            queryVectors.add(vector);
        }

        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);
        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16, // graph degree
                100, // construction search depth
                1.2f, // allow degree overflow during construction by this factor
                1.2f); // relax neighbor diversity requirement by this factor
        graphIndex = graphIndexBuilder.build(ravv);

        transactionCount.set(0);
        totalLatency.set(0);
        latencySamples.clear();
        visitedSamples.clear();
        recallSamples.clear();
        startTime = System.currentTimeMillis();
        scheduler = Executors.newScheduledThreadPool(1);
        totalTransactions = 0;

        scheduler.scheduleAtFixedRate(() -> {
            long elapsed = (System.currentTimeMillis() - startTime) / 1000;
            long count = transactionCount.getAndSet(0);
            double meanLatency = (totalTransactions > 0) ? (double) totalLatency.get() / totalTransactions : 0.0;
            double p999Latency = calculateP999Latency();
            double meanVisited = (totalTransactions > 0) ? (double) visitedSamples.stream().mapToInt(Integer::intValue).sum() / totalTransactions : 0.0;
            double recall = (totalTransactions > 0) ? (double) recallSamples.stream().mapToLong(Long::longValue).sum() / totalTransactions : 0.0;
            tableRepresentation.addEntry(elapsed, count, meanLatency, p999Latency, meanVisited, recall);
        }, 1, 1, TimeUnit.SECONDS);
    }

    private VectorFloat<?> createRandomVector(int dimension) {
        VectorFloat<?> vector = VECTOR_TYPE_SUPPORT.createFloatVector(dimension);
        for (int i = 0; i < dimension; i++) {
            vector.set(i, (float) Math.random());
        }
        return vector;
    }

    private double calculateP999Latency() {
        if (latencySamples.isEmpty()) return 0.0;

        List<Long> sortedLatencies = new ArrayList<>(latencySamples);
        Collections.sort(sortedLatencies);

        int p_count = sortedLatencies.size() / 1000;
        int start_index = Math.max(0, sortedLatencies.size() - p_count);
        int end_index = sortedLatencies.size() - 1;
        List<Long> subList = sortedLatencies.subList(start_index, end_index);
        long total = subList.stream().mapToLong(Long::longValue).sum();
        return (double) total / p_count;
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        graphIndexBuilder.close();
        scheduler.shutdown();
        tableRepresentation.print();
    }

    @Benchmark
    public void testOnHeapRandomVectors(Blackhole blackhole) {
        long start = System.nanoTime();
        var queryVector = SiftSmall.randomVector(originalDimension);
        var searchResult = GraphSearcher.search(queryVector,
                10,                            // number of results
                ravv,                               // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL);                          // valid ordinals to consider
        blackhole.consume(searchResult);
        long duration = System.nanoTime() - start;
        long durationMicro = TimeUnit.NANOSECONDS.toMicros(duration);

        visitedSamples.add(searchResult.getVisitedCount());
        transactionCount.incrementAndGet();
        totalLatency.addAndGet(durationMicro);
        latencySamples.add(durationMicro);
        totalTransactions++;
    }

}
