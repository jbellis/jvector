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

package io.github.jbellis.jvector.example;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;

import static io.github.jbellis.jvector.vector.VectorEncoding.FLOAT32;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Haystack {
    private static void testOneGraph(int nVectors, int M, int efConstruction) throws IOException {
        var R = new Random();
        var L = IntStream.range(0, nVectors).parallel()
                         .mapToObj(i -> new float[] { R.nextFloat(), R.nextFloat() })
                         .collect(Collectors.toList());
        var floatVectors = new ListRandomAccessVectorValues(L, 2);

        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, FLOAT32, EUCLIDEAN, M, efConstruction, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();
        System.out.format("%nBuild nVectors=%d, M=%d ef=%d in %.2fs with %.2f short edges%n",
                nVectors, M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);

        var queryCount = 1000;
        var bits = new FixedBitSet(nVectors);
        for (int i = (int) max(8, nVectors * 0.001); i <= nVectors / 2; i *= 2) {
            while (bits.cardinality() < i) {
                bits.set(R.nextInt(nVectors));
            }
            for (var topK : List.of(5, 10, 20, 30, 50, 75, 100)) {
                if (topK >= i) {
                    break;
                }
                runQueries(queryCount, topK, floatVectors, onHeapGraph, bits);
            }
        }
        runQueries(queryCount, 100, floatVectors, onHeapGraph, null);
    }

    private static void runQueries(int queryCount, int topK, ListRandomAccessVectorValues floatVectors, OnHeapGraphIndex<float[]> onHeapGraph, FixedBitSet bits) {
        var R = new Random();
        int nBitsSet = bits == null ? onHeapGraph.size() : bits.cardinality();
        LongAdder visited = new LongAdder();
        IntStream.range(0, queryCount).parallel().forEach(__ -> {
            var queryVector = new float[] { R.nextFloat(), R.nextFloat() };
            SearchResult sr = GraphSearcher.search(queryVector, topK, floatVectors, FLOAT32, EUCLIDEAN, onHeapGraph, bits);
            visited.add(sr.getVisitedCount());
        });
        if (bits == null) {
            System.out.printf("Looking for top %d of %d ordinals required visiting %d nodes%n", topK, onHeapGraph.size(), visited.sum() / queryCount);
        } else {
            System.out.printf("Looking for top %d of %d ordinals required visiting %d nodes%n", topK, nBitsSet, visited.sum() / queryCount);
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        for (int i = 1024; i <= 1024*1024; i *= 2) {
            testOneGraph(i, 16, 100);
        }
    }
}
