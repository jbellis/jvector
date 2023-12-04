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
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.util.FixedBitSet;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.LongAdder;

import static io.github.jbellis.jvector.vector.VectorEncoding.FLOAT32;
import static io.github.jbellis.jvector.vector.VectorSimilarityFunction.EUCLIDEAN;
import static java.lang.Math.max;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    /**
     * build a graph and generate datapoints for acceptable ordinal counts ranging from
     * 1% of the DataSet size, to 1/2 of the graph size, and for topK from 1 .. 1000
     */
    private static void testOneGraph(int M, int efConstruction, DataSet ds) throws IOException {
        int nVectors = ds.baseVectors.size();
        var floatVectors = new ListRandomAccessVectorValues(ds.baseVectors.subList(0, nVectors), ds.baseVectors.get(0).length);

        var start = System.nanoTime();
        var builder = new GraphIndexBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.2f, 1.4f);
        var onHeapGraph = builder.build();
        var avgShortEdges = onHeapGraph.getAverageShortEdges();
        System.out.format("Build N=%d M=%d ef=%d in %.2fs with %.2f short edges%n",
                floatVectors.size(), M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0, avgShortEdges);

        var queryCount = 1000;
        var bits = new FixedBitSet(nVectors);
        var R = new Random();
        for (int i = (int) max(8, nVectors * 0.01); i <= nVectors / 2; i *= 2) {
            while (bits.cardinality() < i) {
                bits.set(R.nextInt(nVectors));
            }
            for (var topK : List.of(1, 3, 5, 10, 20, 30, 50, 75, 100, 200, 500, 1000)) {
                if (topK >= i) {
                    break;
                }
                runQueries(ds.queryVectors.subList(0, queryCount), topK, floatVectors, onHeapGraph, bits);
            }
        }
        runQueries(ds.queryVectors.subList(0, queryCount), 100, floatVectors, onHeapGraph, Bits.ALL);
    }

    private static void runQueries(List<float[]> queries, int topK, ListRandomAccessVectorValues floatVectors, OnHeapGraphIndex<float[]> onHeapGraph, Bits bits) {
        int nBitsSet = bits instanceof Bits.MatchAllBits ? onHeapGraph.size() : ((FixedBitSet) bits).cardinality();
        LongAdder visited = new LongAdder();
        queries.stream().parallel().forEach(queryVector -> {
            SearchResult sr = GraphSearcher.search(queryVector, topK, floatVectors, FLOAT32, EUCLIDEAN, onHeapGraph, bits);
            visited.add(sr.getVisitedCount());
        });
        System.out.printf("Looking for top %d of %d ordinals required visiting %d nodes%n", topK, nBitsSet, visited.sum() / queries.size());
    }

    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var nwFiles = List.of(
                "intfloat_e5-small-v2_100000",
                "textembedding-gecko_100000",
                "ada_002_100000");
        for (var nwDatasetName : nwFiles) {
            DownloadHelper.maybeDownloadFvecs(nwDatasetName);
            gridSearch(loadNWDataData(nwDatasetName));
        }

        var files = List.of(
                "hdf5/nytimes-256-angular.hdf5",
                "hdf5/glove-100-angular.hdf5",
                "hdf5/glove-200-angular.hdf5",
                "hdf5/sift-128-euclidean.hdf5",
                "hdf5/fashion-mnist-784-euclidean.hdf5");
        for (var f : files) {
            DownloadHelper.maybeDownloadHdf5(f);
            gridSearch(Hdf5Loader.load(f));
        }
    }

    private static DataSet loadNWDataData(String name) throws IOException {
        var path = "wikipedia_squad/100k";
        var baseVectors = SiftLoader.readFvecs("fvec/" + path + "/" + name + "_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("fvec/" + path + "/" + name + "_query_vectors_10000.fvec");
        var gt = SiftLoader.readIvecs("fvec/" + path + "/" + name + "_indices_query_10000.ivec");
        var ds = DataSet.getScrubbedDataSet(name,
                                            VectorSimilarityFunction.DOT_PRODUCT,
                                            baseVectors,
                                            queryVectors,
                                            gt);
        return ds;
    }

    private static void gridSearch(DataSet fullDataSet) throws IOException {
        for (int N = 2048; N <= fullDataSet.baseVectors.size(); N *= 2) {
            var ds = new DataSet(fullDataSet.name + "/" + N,
                                 fullDataSet.similarityFunction,
                                 fullDataSet.baseVectors.subList(0, N),
                                 fullDataSet.queryVectors,
                                 fullDataSet.groundTruth);
            testOneGraph(16, 100, ds);
        }
    }
}
