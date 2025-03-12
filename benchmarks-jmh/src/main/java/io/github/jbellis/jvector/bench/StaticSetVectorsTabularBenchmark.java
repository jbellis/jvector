/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.bench;

import io.github.jbellis.jvector.bench.output.TableRepresentation;
import io.github.jbellis.jvector.bench.output.TextTable;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Fork(1)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@Threads(1)
public class StaticSetVectorsTabularBenchmark {
    private static final Logger log = LoggerFactory.getLogger(StaticSetVectorsTabularBenchmark.class);
    private RandomAccessVectorValues ravv;
    private ArrayList<VectorFloat<?>> baseVectors;
    private ArrayList<VectorFloat<?>> queryVectors;
    private ArrayList<Set<Integer>> groundTruth;
    private GraphIndexBuilder graphIndexBuilder;
    private GraphIndex graphIndex;
    int originalDimension;

    private final AtomicLong transactionCount = new AtomicLong(0);
    private final AtomicLong totalLatency = new AtomicLong(0);
    private final AtomicInteger testCycle = new AtomicInteger(0);
    private final Queue<Long> latencySamples = new ConcurrentLinkedQueue<>(); // Store latencies for P99.9
    private final Queue<Integer> visitedSamples = new ConcurrentLinkedQueue<>();
    private ScheduledExecutorService scheduler;
    private long startTime;

    private final TableRepresentation tableRepresentation = new TextTable(); // Keep TextTable only for now

    @Setup
    public void setup() throws IOException {
        var siftPath = "siftsmall";
        baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
        queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));
        groundTruth = SiftLoader.readIvecs(String.format("%s/siftsmall_groundtruth.ivecs", siftPath));
        log.info("base vectors size: {}, query vectors size: {}, loaded, dimensions {}",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());
        originalDimension = baseVectors.get(0).length();
        // wrap the raw vectors in a RandomAccessVectorValues
        ravv = new ListRandomAccessVectorValues(baseVectors, originalDimension);

        // score provider using the raw, in-memory vectors
        BuildScoreProvider bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.EUCLIDEAN);

        graphIndexBuilder = new GraphIndexBuilder(bsp,
                ravv.dimension(),
                16,  // graph degree
                100,    // construction search depth
                1.2f,   // allow degree overflow during construction by this factor
                1.2f);  // relax neighbor diversity requirement by this factor
        graphIndex = graphIndexBuilder.build(ravv);

        transactionCount.set(0);
        totalLatency.set(0);
        latencySamples.clear();
        startTime = System.currentTimeMillis();
        scheduler = Executors.newScheduledThreadPool(1);

        scheduler.scheduleAtFixedRate(() -> {
            long elapsed = (System.currentTimeMillis() - startTime) / 1000;
            long count = transactionCount.getAndSet(0);
            long latency = totalLatency.getAndSet(0);
            double meanLatency = (count > 0) ? (double) latency / count : 0.0;
            double p999Latency = calculateP999Latency();
            double meanVisited = (count > 0) ? (double) visitedSamples.stream().mapToInt(Integer::intValue).sum() / count : 0.0;

            tableRepresentation.addEntry(elapsed, count, meanLatency, p999Latency, meanVisited);
        }, 1, 1, TimeUnit.SECONDS);
    }

    private double calculateP999Latency() {
        if (latencySamples.isEmpty()) return 0.0;

        List<Long> sortedLatencies = new ArrayList<>(latencySamples);
        Collections.sort(sortedLatencies);

        int index = (int) Math.ceil(sortedLatencies.size() * 0.999) - 1;
        return sortedLatencies.get(Math.max(index, 0));
    }

    @TearDown
    public void tearDown() throws IOException {
        baseVectors.clear();
        queryVectors.clear();
        groundTruth.clear();
        graphIndexBuilder.close();
        scheduler.shutdown();
        tableRepresentation.print();
    }

    @Benchmark
    public void testOnHeapWithRandomQueryVectors(Blackhole blackhole) throws IOException {
        long start = System.nanoTime();
        int offset = testCycle.getAndIncrement() % queryVectors.size();
        var queryVector = queryVectors.get(offset);
        // Your benchmark code here
        var searchResult = GraphSearcher.search(queryVector,
                10, // number of results
                ravv, // vectors we're searching, used for scoring
                VectorSimilarityFunction.EUCLIDEAN, // how to score
                graphIndex,
                Bits.ALL); // valid ordinals to consider
        blackhole.consume(searchResult);
        long duration = System.nanoTime() - start;
        long durationMicro = TimeUnit.NANOSECONDS.toMicros(duration);

        visitedSamples.add(searchResult.getVisitedCount());
        transactionCount.incrementAndGet();
        totalLatency.addAndGet(durationMicro);
        latencySamples.add(durationMicro);
    }

}
