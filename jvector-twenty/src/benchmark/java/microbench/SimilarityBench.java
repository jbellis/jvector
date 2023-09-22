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

import org.openjdk.jmh.annotations.*;


@Warmup(iterations = 2, time = 5)
@Measurement(iterations = 3, time = 10)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = {"--add-modules", "jdk.incubator.vector", "--enable-preview" ,"-XX:+AlignVector"})
public class SimilarityBench {

/*    private static final int SIZE = 1024;
    @State(Scope.Benchmark)
    public static class Parameters {
        MemorySegment fm0 = Arena.global().allocateArray(ValueLayout.JAVA_FLOAT, SIZE);
        MemorySegment fm1 = Arena.global().allocateArray(ValueLayout.JAVA_FLOAT, SIZE);

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
        bh.consume(p.fa0[10]);
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        for (int i = 0; i < SIZE; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector fv0 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, p.fa0, i);
            FloatVector fv1 = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, p.fa1, i);

            acc = acc.add(fv0.mul(fv1));
        }

        bh.consume(acc.reduceLanes(VectorOperators.ADD));
    }

    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void dotProductMemory(Blackhole bh, Parameters p) {
        bh.consume(p.fm0.getAtIndex(ValueLayout.JAVA_FLOAT, 10));
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        for (int i = 0; i < SIZE; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector fv0 = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, p.fm0, i, ByteOrder.LITTLE_ENDIAN);
            FloatVector fv1 = FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, p.fm1, i, ByteOrder.LITTLE_ENDIAN);

            acc = acc.add(fv0.mul(fv1));
        }

        bh.consume(acc.reduceLanes(VectorOperators.ADD));
    }


    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }*/
}
