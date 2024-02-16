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
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--enable-preview", "-Djvector.experimental.enable_native_vectorization=true"})
public class SimilarityBench {

    static VectorFloat<?> A_4 = TestUtil.randomVector(new Random(), 4);
    static VectorFloat<?> B_4 = TestUtil.randomVector(new Random(), 4);
    static VectorFloat<?> A_8 = TestUtil.randomVector(new Random(), 8);
    static VectorFloat<?> B_8 = TestUtil.randomVector(new Random(), 8);
    static VectorFloat<?> A_16 = TestUtil.randomVector(new Random(), 16);
    static VectorFloat<?> B_16 = TestUtil.randomVector(new Random(), 16);


    static

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_4(Blackhole bh) {
        bh.consume(VectorUtil.dotProduct(A_4, B_4));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_8(Blackhole bh) {
        bh.consume(VectorUtil.dotProduct(A_8, B_8));
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void testDotProduct_16(Blackhole bh) {
        bh.consume(VectorUtil.dotProduct(A_16, B_16));
    }




    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}

