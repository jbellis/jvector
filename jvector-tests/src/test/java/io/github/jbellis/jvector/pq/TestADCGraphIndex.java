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
import io.github.jbellis.jvector.disk.SimpleMappedReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.ADCGraphIndex;
import io.github.jbellis.jvector.graph.disk.ADCView;
import io.github.jbellis.jvector.graph.disk.CachingGraphIndex;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static io.github.jbellis.jvector.TestUtil.createRandomVectors;
import static org.junit.Assert.assertEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestADCGraphIndex extends RandomizedTest {

    private Path testDirectory;

    private TestUtil.FullyConnectedGraphIndex fullyConnectedGraph;
    private TestUtil.RandomlyConnectedGraphIndex randomlyConnectedGraph;

    @Before
    public void setup() throws IOException {
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testFusedGraph() throws Exception {
        // generate random graph, M=32, 256-dimension vectors
        var graph = new TestUtil.RandomlyConnectedGraphIndex(50_000, 32, getRandom());
        var outputPath = testDirectory.resolve("large_graph");
        var vectors = createRandomVectors(50000, 64);
        var ravv = new ListRandomAccessVectorValues(vectors, 64);
        var pq = ProductQuantization.compute(ravv, 16, 32, false);
        var compressed = pq.encodeAll(vectors);
        var pqv = new PQVectors(pq, compressed);

        TestUtil.writeFusedGraph(graph, ravv, pqv, outputPath);

        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString());
             var onDiskGraph = ADCGraphIndex.load(marr::duplicate, 0);
             var cachedOnDiskGraph = CachingGraphIndex.from(onDiskGraph))
        {
            TestUtil.assertGraphEquals(graph, onDiskGraph);
            TestUtil.assertGraphEquals(graph, cachedOnDiskGraph);
            try (var cachedOnDiskView = (ADCView) cachedOnDiskGraph.getView())
            {
                var queryVector = TestUtil.randomVector(getRandom(), 64);

                for (var similarity : List.of(VectorSimilarityFunction.DOT_PRODUCT, VectorSimilarityFunction.EUCLIDEAN)) {
                    var fusedScoreFunction = cachedOnDiskView.approximateScoreFunctionFor(queryVector, similarity);
                    var scoreFunction = pqv.precomputedScoreFunctionFor(queryVector, similarity);
                    for (int i = 0; i < 100; i++) {
                        var ordinal = getRandom().nextInt(graph.size());
                        var bulkSimilarities = fusedScoreFunction.edgeLoadingSimilarityTo(ordinal);
                        var neighbors = cachedOnDiskView.getNeighborsIterator(ordinal);
                        for (int j = 0; neighbors.hasNext(); j++) {
                            var neighbor = neighbors.next();
                            assertEquals(scoreFunction.similarityTo(neighbor), bulkSimilarities.get(j), 0.0001);
                        }
                    }
                }
            }
        }
    }
}
