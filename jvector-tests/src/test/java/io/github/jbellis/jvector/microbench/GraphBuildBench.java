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


import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
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

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 10)
@Fork(warmups = 0, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "-XX:+UnlockDiagnosticVMOptions", "--enable-preview", "-XX:+PreserveFramePointer", "-XX:+DebugNonSafepoints"})
public class GraphBuildBench {

    @State(Scope.Benchmark)
    public static class Parameters {
        final DataSet ds;
        final ListRandomAccessVectorValues ravv;

        public Parameters() {
            this.ds = Hdf5Loader.load("hdf5/glove-100-angular.hdf5");
            this.ravv = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testGraphBuild(Blackhole bh, Parameters p) {
        long start = System.nanoTime();
        GraphIndexBuilder<float[]> graphIndexBuilder =  new GraphIndexBuilder<>(p.ravv, VectorEncoding.FLOAT32, p.ds.similarityFunction, 8, 60, 1.2f, 1.4f);
        var onHeapGraph = graphIndexBuilder.build();
        var avgShortEdges = onHeapGraph.getAverageShortEdges();
        System.out.format("Build M=%d ef=%d in %.2fs with %.2f short edges%n",
                32, 600, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);
    }
}

