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
package io.github.jbellis.jvector.microbench;

import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;

@State(Scope.Benchmark)
@Warmup(iterations = 5)
@Measurement(iterations = 10)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(warmups = 0, value = 1, jvmArgsAppend = {
        "-XX:+UseG1GC",
        "--add-opens",
        "java.base/java.nio=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
        "--enable-preview",
})
public class PQBenchmark {

    @Param("512")
    public int dimension;

    @Param("200000")
    public int size;

    @Param("4")
    public int pqFactor;

    @Param("32")
    public int core;

    private Random random;
    private ProductQuantization pq;
    private List<float[]> floats;
    private float[] queryTestVector;
    private ForkJoinPool fjp;
    private PQVectors heapPqVectors;
    private PQVectors offHeapPqVectors;

    @Setup(Level.Trial)
    public void setUp() {
        long startTime = System.nanoTime();
        fjp = new ForkJoinPool(core);
        Random random = new Random();
        random.setSeed(0);
        floats = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            floats.add(TestUtil.randomVector(random, dimension));
        }
        int pqDims = dimension / pqFactor;
        pq = ProductQuantization.compute(new ListRandomAccessVectorValues(floats, dimension), pqDims, false, fjp, fjp);
        System.out.println("setup use time: " + (System.nanoTime() - startTime) / 1000000.0 + "ms");
        byte[][] compressedVectors = pq.encodeAll(floats, fjp);
        heapPqVectors = new PQVectors(pq, compressedVectors);
        offHeapPqVectors = new PQVectors(pq, toDirectByteBuffer(compressedVectors), size);
        queryTestVector = TestUtil.randomVector(random, dimension);
    }

    private static ByteBuffer toDirectByteBuffer(byte[][] compressedVectors) {
        ByteBuffer cv = ByteBuffer.allocateDirect(compressedVectors.length * compressedVectors[0].length).order(ByteOrder.LITTLE_ENDIAN);
        for (byte[] bytes : compressedVectors) {
            cv.put(bytes);
        }
        cv.flip();
        return cv;
    }

    @Benchmark
    public byte[][] buildOnHeap() {
        return pq.encodeAll(floats, fjp);
    }

    @Benchmark
    public ByteBuffer buildOffHeap() {
        return pq.encodeAllToOffHeap(floats, fjp);
    }

    @Benchmark
    public float scoreHeap() {
        float maxScore = -1;
        var scoreFunction = heapPqVectors.approximateScoreFunctionFor(queryTestVector, EUCLIDEAN);
        for (int i = 0; i < size; i++) {
            float score = scoreFunction.similarityTo(i);
            if (score > maxScore) {
                maxScore = score;
            }
        }
        return maxScore;
    }

    @Benchmark
    public float scoreOffHeap() {
        float maxScore = -1;
        var scoreFunction = offHeapPqVectors.approximateScoreFunctionFor(queryTestVector, EUCLIDEAN);
        for (int i = 0; i < size; i++) {
            float score = scoreFunction.similarityTo(i);
            if (score > maxScore) {
                maxScore = score;
            }
        }
        return maxScore;
    }

    public static void main(String[] args) throws IOException {
        org.openjdk.jmh.Main.main(new String[]{PQBenchmark.class.getName()});
    }
}
