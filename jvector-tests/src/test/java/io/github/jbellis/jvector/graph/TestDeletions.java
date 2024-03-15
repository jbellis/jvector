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
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.TestUtil.assertGraphEquals;
import static io.github.jbellis.jvector.TestUtil.openFileForWriting;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectors;
import static io.github.jbellis.jvector.graph.TestVectorGraph.createRandomFloatVectorsParallel;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class TestDeletions extends LuceneTestCase {
    @Test
    public void testMarkDeleted() {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete a random entry
        int n = getRandom().nextInt(ravv.size());
        builder.markNodeDeleted(n);
        // check that searching for random vectors never results in the deleted one
        for (int i = 0; i < 100; i++) {
            var v = TestUtil.randomVector(getRandom(), dimension);
            var results = GraphSearcher.search(v, 3, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
            for (var ns : results.getNodes()) {
                assertNotEquals(n, ns.node);
            }
        }
        // check that asking for the entire graph back still doesn't surface the deleted one
        var v = ravv.vectorValue(n);
        var results = GraphSearcher.search(v, ravv.size(), ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(GraphIndex.prettyPrint(graph), ravv.size() - 1, results.getNodes().length);
        for (var ns : results.getNodes()) {
            assertNotEquals(n, ns.node);
        }
    }

    @Test
    public void testCleanup() throws IOException {
        // graph of 10 vectors
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // delete all nodes that connect to a random node
        int nodeToIsolate = getRandom().nextInt(ravv.size());
        int nDeleted = 0;
        try (var view = graph.getView()) {
            for (var i = 0; i < view.size(); i++) {
                for (var it = view.getNeighborsIterator(i); it.hasNext(); ) {
                    if (nodeToIsolate == it.nextInt()) {
                        builder.markNodeDeleted(i);
                        nDeleted++;
                        break;
                    }
                }
            }
        }
        assertNotEquals(0, nDeleted);

        // cleanup removes the deleted nodes
        builder.cleanup();
        assertEquals(ravv.size() - nDeleted, graph.size());

        // cleanup should have added new connections to the node that would otherwise have been disconnected
        var v = ravv.vectorValue(nodeToIsolate).copy();
        var results = GraphSearcher.search(v, 10, ravv, VectorSimilarityFunction.COSINE, graph, Bits.ALL);
        assertEquals(nodeToIsolate, results.getNodes()[0].node);

        // check that we can save and load the graph with "holes" from the deletion
        var testDirectory = Files.createTempDirectory(this.getClass().getSimpleName());
        var outputPath = testDirectory.resolve("on_heap_graph");
        try (var out = openFileForWriting(outputPath)) {
            graph.save(out);
            out.flush();
        }
        var b2 = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        try (var marr = new SimpleMappedReader(outputPath.toAbsolutePath().toString())) {
            b2.load(marr);
        }
        var reloadedGraph = b2.getGraph();
        assertGraphEquals(graph, reloadedGraph);
    }

    @Test
    public void testMarkingAllNodesAsDeleted() {
        // build graph
        int dimension = 2;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectors(10, dimension, getRandom()));
        var builder = new GraphIndexBuilder(ravv, VectorSimilarityFunction.COSINE, 2, 10, 1.0f, 1.0f);
        var graph = TestUtil.buildSequentially(builder, ravv);

        // mark all deleted
        for (var i = 0; i < graph.size(); i++) {
            graph.markDeleted(i);
        }

        // removeDeletedNodes should leave the graph empty
        builder.removeDeletedNodes();
        assertEquals(0, graph.size());
        assertEquals(OnHeapGraphIndex.NO_ENTRY_POINT, graph.entry());
    }

    @Test
    public void testConcurrentDeletes() {
        int dimension = 2;
        var count = 100_000;
        var ravv = MockVectorValues.fromValues(createRandomFloatVectorsParallel(count, dimension));
        var bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VectorSimilarityFunction.DOT_PRODUCT);
        var builder = new GraphIndexBuilder(bsp, 2, 32, 10, 1.0f, 1.0f,
                                            PhysicalCoreExecutor.pool(),
                                            new ForkJoinPool(Runtime.getRuntime().availableProcessors()));

        // need to use a non-reentrant lock because otherwise the thread scheduler thinks that
        // the worker thread executing removeDeletedNodes is available for more tasks once it calls
        // .join on its own parallel task submission. A reentrant lock would allow that thread to
        // start executing removeDeletedNodes a second time, invaliding its assumptions that it does
        // not have to worry about concurrent removals.
        var deleteLock = new NonReentrantLock();
        IntStream.range(0, count).parallel().forEach(i -> {
            var R = ThreadLocalRandom.current();
            if (R.nextDouble() < 0.6) {
                builder.addGraphNode(i, ravv.vectorValue(i));
            } else if (R.nextDouble() < 0.5) {
                GraphSearcher.search(TestUtil.randomVector(R, dimension), 10, ravv, VectorSimilarityFunction.DOT_PRODUCT, builder.getGraph(), Bits.ALL);
            } else if (R.nextDouble() < 0.9) {
                var n = builder.randomLiveNode();
                if (n >= 0) {
                    builder.markNodeDeleted(n);
                }
            } else {
                if (deleteLock.tryLock()) {
                    try {
                        builder.removeDeletedNodes();
                    } finally {
                        deleteLock.unlock();
                    }
                }
            }
        });
        builder.removeDeletedNodes();
        builder.validateAllNodesLive();
    }

    /**
     * A non-reentrant lock for testConcurrentDeletes.
     */
    private static class NonReentrantLock implements Lock {
        private final Semaphore sem = new Semaphore(1);

        @Override
        public void lock() {
            sem.acquireUninterruptibly();
        }

        @Override
        public void lockInterruptibly() throws InterruptedException {
            sem.acquire();
        }

        @Override
        public boolean tryLock() {
            return sem.tryAcquire();
        }

        @Override
        public boolean tryLock(long time, TimeUnit unit) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void unlock() {
            sem.release();
        }

        @Override
        public Condition newCondition() {
            throw new UnsupportedOperationException();
        }
    }
}
