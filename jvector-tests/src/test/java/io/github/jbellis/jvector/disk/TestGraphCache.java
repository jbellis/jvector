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

package io.github.jbellis.jvector.disk;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import io.github.jbellis.jvector.TestUtil;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.writeGraph;
import static org.junit.Assert.*;

public class TestGraphCache extends RandomizedTest {
    private Path testDirectory;
    private Path onDiskGraphIndexPath;
    private RandomAccessVectorValues<float[]> vectors;


    @Before
    public void setup() throws IOException {
        var fullyConnectedGraph = new TestUtil.FullyConnectedGraphIndex<float[]>(0, 6);
        vectors = new ListRandomAccessVectorValues(IntStream.range(0, 6).mapToObj(i -> TestUtil.randomVector(getRandom(), 2)).collect(Collectors.toList()), 2);
        testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        onDiskGraphIndexPath = testDirectory.resolve("fullyConnectedGraph");
        writeGraph(fullyConnectedGraph, vectors, onDiskGraphIndexPath);
    }

    @After
    public void tearDown() {
        TestUtil.deleteQuietly(testDirectory);
    }

    @Test
    public void testGraphCacheLoading() throws Exception {
        try (var marr = new SimpleMappedReader(onDiskGraphIndexPath.toAbsolutePath().toString());
             var onDiskGraph = new OnDiskGraphIndex<float[]>(marr::duplicate, 0))
        {
            var none = GraphCache.load(onDiskGraph, -1);
            assertEquals(0, none.ramBytesUsed());
            assertNull(none.getNode(0));
            var zero = GraphCache.load(onDiskGraph, 0);
            assertNotNull(zero.getNode(0));
            assertNull(zero.getNode(1));
            var one = GraphCache.load(onDiskGraph, 1);
            // move from caching entry node to entry node + all its neighbors (5)
            assertEquals(one.ramBytesUsed(), zero.ramBytesUsed() * (onDiskGraph.size()));
            for (int i = 0; i < 6; i++) {
                assertArrayEquals(one.getNode(i).vector, vectors.vectorValue(i), 0);
                // fully connected,
                assertEquals(one.getNode(i).neighbors.length, onDiskGraph.maxDegree());
            }
        }
    }
}