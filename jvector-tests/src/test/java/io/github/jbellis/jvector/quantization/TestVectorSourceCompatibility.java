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

package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.disk.FvecFileVectorValues;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.vector.*;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ForkJoinPool;

import static org.junit.Assert.*;

/** Compare the same quantizer on file, resident and borrowed point-only sources. */
public class TestVectorSourceCompatibility {
    private static final int ROWS = 257;
    private static final int[] DIMENSIONS = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 768, 1024, 1536, 3072};

    private static final class Input implements AutoCloseable {
        final Path path;
        final List<VectorFloat<?>> vectors = new ArrayList<>();
        final ListRandomAccessVectorValues resident;
        final FvecFileVectorValues file;
        Input(int dimension) throws Exception {
            path = Files.createTempFile("source-compatibility", ".fvecs");
            var bytes = ByteBuffer.allocate(ROWS * (dimension + 1) * 4).order(ByteOrder.LITTLE_ENDIAN);
            var random = new Random(42);
            var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
            for (int row = 0; row < ROWS; row++) {
                var vector = vts.createFloatVector(dimension);
                for (int col = 0; col < dimension; col++) vector.set(col, random.nextFloat() * 2 - 1);
                VectorUtil.l2normalize(vector); // anisotropic training requires unit vectors
                bytes.putInt(dimension);
                for (int col = 0; col < dimension; col++) bytes.putFloat(vector.get(col));
                vectors.add(vector);
            }
            Files.write(path, bytes.array());
            resident = new ListRandomAccessVectorValues(vectors, dimension);
            file = FvecFileVectorValues.open(path, FvecFileVectorValues.Options.defaults()
                    .withIoThreads(2).withBatchVectors(7).withReadAhead(3));
        }
        RandomAccessVectorValues pointOnly() { return new Borrowed(resident); }
        public void close() throws Exception { try { file.close(); } finally { Files.delete(path); } }
    }

    /** Models existing integrations that supply one borrowed vector per view, without hooks. */
    private static final class Borrowed implements RandomAccessVectorValues {
        final RandomAccessVectorValues source;
        final VectorFloat<?> scratch;
        Borrowed(RandomAccessVectorValues source) {
            this.source = source;
            scratch = VectorizationProvider.getInstance().getVectorTypeSupport().createFloatVector(source.dimension());
        }
        public int size() { return source.size(); }
        public int dimension() { return source.dimension(); }
        public boolean isValueShared() { return true; }
        public RandomAccessVectorValues copy() { return new Borrowed(source); }
        public VectorFloat<?> getVector(int ordinal) {
            source.getVectorInto(ordinal, scratch, 0);
            return scratch;
        }
    }

    private static void encodingMatches(ProductQuantization pq, Input input, ForkJoinPool pool) {
        var expected = pq.encodeAll(input.resident, pool);
        assertEquals(expected, pq.encodeAll(input.file, pool));
        assertEquals(expected, pq.encodeAll(input.pointOnly(), pool));
        for (int i = 0; i < ROWS; i++) assertEquals(pq.encode(input.vectors.get(i)), expected.get(i));
        assertScoresEqual(expected, pq.encodeAll(input.file, pool), input.vectors.get(0));
    }

    private static void assertScoresEqual(CompressedVectors a, CompressedVectors b, VectorFloat<?> query) {
        for (var metric : VectorSimilarityFunction.values()) {
            var scoreA = a.scoreFunctionFor(query.copy(), metric);
            var scoreB = b.scoreFunctionFor(query.copy(), metric);
            var tableA = a.precomputedScoreFunctionFor(query.copy(), metric);
            var tableB = b.precomputedScoreFunctionFor(query.copy(), metric);
            for (int i = 0; i < a.count(); i++) {
                assertEquals(scoreA.similarityTo(i), scoreB.similarityTo(i), 0f);
                assertEquals(tableA.similarityTo(i), tableB.similarityTo(i), 0f);
            }
        }
    }

    @Test(timeout = 180000)
    public void pqWidthsCenteringAndClusterCounts() throws Exception {
        var pool = new ForkJoinPool(3);
        try {
            // Widths on both sides of SIMD boundaries and uneven final subspaces.
            for (int dimension : DIMENSIONS) {
                try (var input = new Input(dimension)) {
                    for (boolean centered : new boolean[] {false, true}) {
                        int m = Math.max(1, dimension / 8);
                        int k = dimension == 8 ? 256 : 16;
                        var pq = ProductQuantization.compute(input.resident, m, k, centered,
                                io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED, pool, pool);
                        encodingMatches(pq, input, pool);
                    }
                }
            }
        } finally { pool.shutdown(); }
    }

    @Test(timeout = 180000)
    public void anisotropicTrainingAndRefinementUseTheSameSamples() throws Exception {
        var pool = new ForkJoinPool(3);
        try (var input = new Input(17)) {
            assertEquals(ProductQuantization.extractTrainingVectors(input.resident, pool),
                    ProductQuantization.extractTrainingVectors(input.file, pool));
            assertEquals(ProductQuantization.extractTrainingVectors(input.resident, pool),
                    ProductQuantization.extractTrainingVectors(input.pointOnly(), pool));
            encodingMatches(ProductQuantization.compute(input.file, 3, 31, true, -1f, pool, pool), input, pool);
            var pq = ProductQuantization.compute(input.file, 3, 16, false, 0.2f, pool, pool);
            encodingMatches(pq, input, pool);
            var refined = pq.refine(input.file, 1, 0.2f, pool, pool);
            encodingMatches(refined, input, pool);
        } finally { pool.shutdown(); }
    }

    @Test(timeout = 180000)
    public void nvqUnevenSubvectorsAndLearningModes() throws Exception {
        var pool = new ForkJoinPool(3);
        try {
            for (int dimension : DIMENSIONS) {
                try (var input = new Input(dimension)) {
                    for (int parts : new int[] {1, Math.min(3, dimension)}) {
                        var nvq = NVQuantization.compute(input.resident, parts);
                        assertEquals(nvq, NVQuantization.compute(input.file, parts));
                        assertEquals(nvq, NVQuantization.compute(input.pointOnly(), parts));
                        for (boolean learn : new boolean[] {false, true}) {
                            nvq.learn = learn;
                            var expected = nvq.encodeAll(input.resident, pool);
                            assertEquals(expected, nvq.encodeAll(input.file, pool));
                            assertEquals(expected, nvq.encodeAll(input.pointOnly(), pool));
                            assertScoresEqual(expected, nvq.encodeAll(input.file, pool), input.vectors.get(0));
                        }
                    }
                }
            }
        } finally { pool.shutdown(); }
    }

    @Test(timeout = 60000)
    public void binaryEncodingAndFutureScanConsumerWorkWithBothSourceContracts() throws Exception {
        var pool = new ForkJoinPool(3);
        try (var input = new Input(65)) {
            var bq = new BinaryQuantization(65);
            assertEquals(bq.encodeAll(input.resident, pool), bq.encodeAll(input.file, pool));
            assertEquals(bq.encodeAll(input.resident, pool), bq.encodeAll(input.pointOnly(), pool));
            for (var source : List.of(input.resident, input.file, input.pointOnly())) {
                // A future algorithm needs only this source-independent scan, not file type checks.
                var copied = new VectorFloat<?>[ROWS];
                VectorAccess.forEach(source, pool, (ordinal, vector) -> copied[ordinal] = vector.copy());
                assertEquals(input.vectors, Arrays.asList(copied));
            }
        } finally { pool.shutdown(); }
    }

    @Test(timeout = 60000)
    public void mappedPqEncodingPreservesOutputPositionsAndDuplicates() throws Exception {
        var pool = new ForkJoinPool(3);
        try (var input = new Input(17)) {
            var pq = ProductQuantization.compute(input.resident, 3, 16, false,
                    io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED, pool, pool);
            int[] ordinals = {256, 5, 256, 0, 3, 1};
            for (var source : List.of(input.file, input.pointOnly())) {
                var encoded = PQVectors.encodeAndBuild(pq, ordinals.length, i -> ordinals[i], source, pool);
                for (int i = 0; i < ordinals.length; i++)
                    assertEquals(pq.encode(input.vectors.get(ordinals[i])), encoded.get(i));
            }
            var empty = new ListRandomAccessVectorValues(Collections.emptyList(), 17);
            assertThrows(IllegalArgumentException.class, () -> pq.encodeAll(empty, pool));
            assertEquals(0, new BinaryQuantization(17).encodeAll(empty, pool).count());
            assertEquals(0, NVQuantization.compute(input.resident, 1).encodeAll(empty, pool).count());
        } finally { pool.shutdown(); }
    }

    @Test public void nvqMissingOrdinalsEncodeAsZeroVectors() {
        var vts = VectorizationProvider.getInstance().getVectorTypeSupport();
        var first = vts.createFloatVector(new float[] {1, 2, 3});
        var last = vts.createFloatVector(new float[] {3, 2, 1});
        var sparse = new ListRandomAccessVectorValues(Arrays.asList(first, null, last), 3);
        var dense = new ListRandomAccessVectorValues(List.of(first, last), 3);
        var pool = new ForkJoinPool(2);
        BatchedVectorValues batched = new BatchedVectorValues() {
            public int size() { return sparse.size(); }
            public int dimension() { return sparse.dimension(); }
            public boolean isValueShared() { return false; }
            public RandomAccessVectorValues copy() { return this; }
            public VectorFloat<?> getVector(int ordinal) { return sparse.getVector(ordinal); }
            public VectorCursor openRange(int start, int end) { return VectorAccess.openRange(sparse, start, end); }
            public VectorCursor openSelection(int[] ordinals, int offset, int count) {
                return VectorAccess.openSelection(sparse, ordinals, offset, count);
            }
        };
        try {
            var nvq = NVQuantization.compute(dense, 2);
            var zero = vts.createFloatVector(3);
            for (boolean learn : new boolean[] {false, true}) {
                nvq.learn = learn;
                for (var source : List.of(sparse, batched)) {
                    var encoded = nvq.encodeAll(source, pool);
                    assertEquals(3, encoded.count());
                    assertEquals(nvq.encode(first), encoded.get(0));
                    assertEquals(nvq.encode(zero), encoded.get(1));
                    assertEquals(nvq.encode(last), encoded.get(2));
                }
            }
        } finally { pool.shutdown(); }
    }
}
