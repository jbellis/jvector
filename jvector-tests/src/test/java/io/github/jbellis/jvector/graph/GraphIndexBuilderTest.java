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

package io.github.jbellis.jvector.graph;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Supplier;

import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class GraphIndexBuilderTest extends LuceneTestCase {

    private Path testDirectory;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testSaveAndLoad() throws IOException {
        int dimension = randomIntBetween(2, 32);
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(randomIntBetween(10, 100), dimension, getRandom()));
        Supplier<GraphIndexBuilder> newBuilder = () ->
            new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);

        var indexDataPath = testDirectory.resolve("index_builder.data");
        var builder = newBuilder.get();

        try (var graph = TestUtil.buildSequentially(builder, ravv);
             var out = TestUtil.openFileForWriting(indexDataPath)) {
            graph.save(out);
            out.flush();
        }

        builder = newBuilder.get();
        try(var reader = new SimpleMappedReader(indexDataPath)) {
            builder.load(reader);
        }

        assertEquals(ravv.size(), builder.graph.size());
        for (int i = 0; i < ravv.size(); i++) {
            assertTrue(builder.graph.containsNode(i));
        }
    }
}
