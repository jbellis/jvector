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


import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1)
public class SimilarityBench {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    static int SIZE = 256;
    static final VectorFloat<?> q1 = vectorTypeSupport.createFloatType(SIZE);
    static final VectorFloat<?> q2 = vectorTypeSupport.createFloatType(SIZE);

    static final VectorFloat<?> q3 = vectorTypeSupport.createFloatType(SIZE);

    static {
        for (int i = 0; i < q1.length(); i++) {
            q1.set(i, ThreadLocalRandom.current().nextFloat());
            q2.set(i, ThreadLocalRandom.current().nextFloat());
        }

        q3.set(0, ThreadLocalRandom.current().nextFloat());
        q3.set(1, ThreadLocalRandom.current().nextFloat());
    }

    @State(Scope.Benchmark)
    public static class Parameters {

    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProduct(Blackhole bh, Parameters p) {
        bh.consume(VectorUtil.dotProduct(q3, 0, q1, 22, q3.length()));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
