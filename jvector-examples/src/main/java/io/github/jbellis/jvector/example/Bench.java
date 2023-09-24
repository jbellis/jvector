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

import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.vector.VectorEncoding.FLOAT32;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static java.lang.Math.max;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    private static void testOneGraph(int nVectors, int M, int efConstruction, DataSet ds) throws IOException {
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors.subList(0, nVectors), ds.baseVectors.get(0).length);

        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        var avgShortEdges = IntStream.range(0, onHeapGraph.size()).mapToDouble(i -> onHeapGraph.getNeighbors(i).getShortEdges()).average().orElseThrow();
        System.out.format("Build N=%d M=%d ef=%d in %.2fs with %.2f short edges%n",
                floatVectors.size(), M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);

        var queryCount = 1000;
        var bits = new FixedBitSet(nVectors);
        var R = new Random();
        for (int i = (int) max(8, nVectors * 0.001); i <= nVectors / 2; i *= 2) {
            while (bits.cardinality() < i) {
                bits.set(R.nextInt(nVectors));
            }
            for (var topK : List.of(5, 10, 20, 30, 50, 75, 100)) {
                if (topK >= i) {
                    break;
                }
                runQueries(ds.queryVectors.subList(0, queryCount), topK, floatVectors, onHeapGraph, bits);
            }
        }
        runQueries(ds.queryVectors.subList(0, queryCount), 100, floatVectors, onHeapGraph, null);
    }

    private static void runQueries(List<float[]> queries, int topK, ListRandomAccessVectorValues floatVectors, OnHeapGraphIndex<float[]> onHeapGraph, FixedBitSet bits) {
        int nBitsSet = bits == null ? onHeapGraph.size() : bits.cardinality();
        LongAdder visited = new LongAdder();
        queries.stream().parallel().forEach(queryVector -> {
            SearchResult sr = GraphSearcher.search(queryVector, topK, floatVectors, FLOAT32, EUCLIDEAN, onHeapGraph, bits);
            visited.add(sr.getVisitedCount());
        });
        if (bits == null) {
            System.out.printf("Looking for top %d of %d ordinals required visiting %d nodes%n", topK, onHeapGraph.size(), visited.sum() / queries.size());
        } else {
            System.out.printf("Looking for top %d of %d ordinals required visiting %d nodes%n", topK, nBitsSet, visited.sum() / queries.size());
        }
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(8, 12, 16, 24, 32, 48, 64);
        var efConstructionGrid = List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchGrid = List.of(1, 2);
        var diskGrid = List.of(false, true);

        // this dataset contains more than 10k query vectors, so we limit it with .subList
        var adaSet = loadWikipediaData();
        gridSearch(adaSet, mGrid, efConstructionGrid, diskGrid, efSearchGrid);

        var files = List.of(
                // large files not yet supported
                // "hdf5/deep-image-96-angular.hdf5",
                // "hdf5/gist-960-euclidean.hdf5",
                "hdf5/nytimes-256-angular.hdf5",
                "hdf5/glove-100-angular.hdf5",
                "hdf5/glove-200-angular.hdf5",
                "hdf5/sift-128-euclidean.hdf5");
        for (var f : files) {
            gridSearch(Hdf5Loader.load(f), mGrid, efConstructionGrid, diskGrid, efSearchGrid);
        }

        // tiny dataset, don't waste time building a huge index
        files = List.of("hdf5/fashion-mnist-784-euclidean.hdf5");
        mGrid = List.of(8, 12, 16, 24);
        efConstructionGrid = List.of(40, 60, 80, 100, 120, 160);
        for (var f : files) {
            gridSearch(Hdf5Loader.load(f), mGrid, efConstructionGrid, diskGrid, efSearchGrid);
        }
    }

    private static DataSet loadWikipediaData() throws IOException {
        var baseVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("fvec/pages_ada_002_100k_query_vectors_10k.fvec").subList(0, 10_000);
        var gt = SiftLoader.readIvecs("fvec/pages_ada_002_100k_indices_query_vectors_10k.ivec").subList(0, 10_000);
        var ds = new DataSet("wikipedia",
                             VectorSimilarityFunction.DOT_PRODUCT,
                             baseVectors,
                             queryVectors,
                             gt);
        System.out.format("%nWikipedia: %d base and %d query vectors loaded, dimensions %d%n",
                          baseVectors.size(), queryVectors.size(), baseVectors.get(0).length);
        return ds;
    }

    private static void gridSearch(DataSet ds, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Boolean> diskOptions, List<Integer> efSearchFactor) throws IOException {
        for (int i = 1024; i < ds.baseVectors.size(); i *= 2) {
            testOneGraph(i, 16, 100, ds);
        }
        testOneGraph(ds.baseVectors.size(), 16, 100, ds);
    }
}
