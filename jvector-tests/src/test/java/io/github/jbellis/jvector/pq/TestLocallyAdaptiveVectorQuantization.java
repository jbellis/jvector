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

package io.github.jbellis.jvector.pq;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.LVQPackedVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestLocallyAdaptiveVectorQuantization extends RandomizedTest {
    private Path testDirectory;
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testLVQSimilarity() throws IOException {
        for (int i = 0; i < 4; i++) {
            testLvqSimilarityOnce();
        }
    }

    public void testLvqSimilarityOnce() throws IOException {
        var dimension = 32 + getRandom().nextInt(1505);
        var randomVectors = TestUtil.createRandomVectors(1000, dimension);
        var ravv = new ListRandomAccessVectorValues(randomVectors, dimension);
        var lvq = LocallyAdaptiveVectorQuantization.compute(ravv);
        var query = TestUtil.randomVector(getRandom(), dimension);
        var lvqView = new MockLVQPackedVectors(lvq, randomVectors, testDirectory);
        float maxDelta = 0;

        for (VectorSimilarityFunction vsf : VectorSimilarityFunction.values()) {
            var lvqSF = lvq.scoreFunctionFrom(query, vsf, lvqView);
            for (int i = 0; i < 1000; i++) {
                var lvqSimilarity = lvqSF.similarityTo(i);
                var actualSimilarity = vsf.compare(query, randomVectors.get(i));
                maxDelta = Math.max(maxDelta, Math.abs(lvqSimilarity - actualSimilarity));
                Assert.assertEquals(actualSimilarity, lvqSimilarity, 0.005);
            }
        }
    }

    private static class MockLVQPackedVectors implements LVQPackedVectors, AutoCloseable {
        private final RandomAccessReader reader;
        private final ByteSequence<?> packedVector;
        private final int encodedVectorSize;

        MockLVQPackedVectors(LocallyAdaptiveVectorQuantization lvq, List<VectorFloat<?>> vectors, Path testDirectory) throws IOException {
            var encodedVectors = lvq.encodeAll(vectors);
            var lvqPath = testDirectory.resolve("lvq" + System.nanoTime());
            try (var out = TestUtil.openDataOutputStream(lvqPath)) {
                for (var encodedVector : encodedVectors) {
                    encodedVector.writePacked(out);
                }
            }
            var dimension = lvq.globalMean.length();
            this.reader = new SimpleMappedReader(lvqPath);
            this.encodedVectorSize = 2 * Float.BYTES + ((dimension % 64 == 0) ? dimension : ((dimension / 64 + 1) * 64));
            this.packedVector = vectorTypeSupport.createByteSequence(encodedVectorSize - 2 * Float.BYTES);
        }

        @Override
        public LocallyAdaptiveVectorQuantization.PackedVector getPackedVector(int ordinal) {
            try {
                reader.seek(ordinal * (long) encodedVectorSize);
                var bias = reader.readFloat();
                var scale = reader.readFloat();
                vectorTypeSupport.readByteSequence(reader, packedVector);
                return new LocallyAdaptiveVectorQuantization.PackedVector(packedVector, bias, scale);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void close() throws Exception {
            reader.close();
        }
    }
}
