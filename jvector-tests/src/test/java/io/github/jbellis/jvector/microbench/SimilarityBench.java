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
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
@State(Scope.Thread)
public class SimilarityBench {

    @Param({"4", "8", "16", "1024"})
    int size = 1024;

    VectorFloat<?> A, B;

    @Setup(Level.Trial)
    public void setUp()
    {
        A = TestUtil.randomVector(new Random(), size);
        B = TestUtil.randomVector(new Random(), size);
    }

    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    @Benchmark
    @Threads(8)
    public void testDotProduct8(Blackhole bh) {
        bh.consume(VectorUtil.dotProduct(A, B));
    }


    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    @Benchmark
    @Threads(1)
    public void testDotProduct1(Blackhole bh) {
        bh.consume(VectorUtil.dotProduct(A, B));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}

