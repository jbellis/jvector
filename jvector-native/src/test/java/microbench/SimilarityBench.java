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

package microbench;

import io.github.jbellis.jvector.vector.cnative.NativeSimdOps;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector", "--enable-preview" ,"-XX:+AlignVector", "-Djava.library.path=/home/jake/workspace/jvector/jvector-native/src/main/c"})
public class SimilarityBench {

    private static final int SIZE = 2;
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_64;
    @State(Scope.Benchmark)
    public static class Parameters {
        MemorySegment fm0 = MemorySegment.ofBuffer(ByteBuffer.allocateDirect(SIZE * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN));
        MemorySegment fm1 = MemorySegment.ofBuffer(ByteBuffer.allocateDirect(SIZE * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN));

        float[] fa0 = new float[SIZE];
        float[] fa1 = new float[SIZE];

        public Parameters(){
            for (int i = 0; i < SIZE; i++) {
                float f0 = ThreadLocalRandom.current().nextFloat();
                float f1 = ThreadLocalRandom.current().nextFloat();

                fa0[i] = f0;
                fa1[i] = f1;

                fm0.setAtIndex(ValueLayout.JAVA_FLOAT, i, f0);
                fm1.setAtIndex(ValueLayout.JAVA_FLOAT, i, f1);
            }
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProductArray(Blackhole bh, Parameters p) {
        FloatVector acc = FloatVector.zero(SPECIES);
        for (int i = 0; i < SIZE; i += SPECIES.length()) {
            FloatVector fv0 = FloatVector.fromArray(SPECIES, p.fa0, i);
            FloatVector fv1 = FloatVector.fromArray(SPECIES, p.fa1, i);

            acc = acc.add(fv0.mul(fv1));
        }

        bh.consume(acc.reduceLanes(VectorOperators.ADD));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProductMemory(Blackhole bh, Parameters p) {
        FloatVector acc = FloatVector.zero(SPECIES);
        for (int i = 0; i < SIZE; i += SPECIES.length()) {
            FloatVector fv0 = FloatVector.fromMemorySegment(SPECIES, p.fm0, i, ByteOrder.LITTLE_ENDIAN);
            FloatVector fv1 = FloatVector.fromMemorySegment(SPECIES, p.fm1, i, ByteOrder.LITTLE_ENDIAN);

            acc = acc.add(fv0.mul(fv1));
        }

        bh.consume(acc.reduceLanes(VectorOperators.ADD));
    }


    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProductNative(Blackhole bh, Parameters p) {
        bh.consume(NativeSimdOps.dot_product_f32(FloatVector.SPECIES_PREFERRED.vectorBitSize(), p.fm0, 0, p.fm1, 0, SIZE));
    }


    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
