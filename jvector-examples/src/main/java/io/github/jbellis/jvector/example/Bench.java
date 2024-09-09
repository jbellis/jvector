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

import io.github.jbellis.jvector.example.util.CompressorParameters;
import io.github.jbellis.jvector.example.util.CompressorParameters.PQParameters;
import io.github.jbellis.jvector.example.util.DataSet;
import io.github.jbellis.jvector.example.util.DataSetCreator;
import io.github.jbellis.jvector.example.util.DownloadHelper;
import io.github.jbellis.jvector.example.util.Hdf5Loader;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.FeatureId;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.pq.KMeansPlusPlusClusterer.UNWEIGHTED;

/**
 * Tests GraphIndexes against vectors from various datasets
 */
public class Bench {
    public static void main(String[] args) throws IOException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(32); // List.of(16, 24, 32, 48, 64, 96, 128);
        var efConstructionGrid = List.of(100); // List.of(60, 80, 100, 120, 160, 200, 400, 600, 800);
        var efSearchGrid = List.of(1.0, 2.0);
        List<Function<DataSet, CompressorParameters>> buildCompression = Arrays.asList(
                ds -> new PQParameters(ds.getDimension() / 8, 256, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN, UNWEIGHTED),
                __ -> CompressorParameters.NONE
        );
        List<Function<DataSet, CompressorParameters>> searchCompression = Arrays.asList(
                __ -> CompressorParameters.NONE,
                // ds -> new CompressorParameters.BQParameters(),
                ds -> new PQParameters(ds.getDimension() / 8, 256, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN, UNWEIGHTED)
        );
        List<EnumSet<FeatureId>> featureSets = Arrays.asList(
                EnumSet.of(FeatureId.INLINE_VECTORS),
                EnumSet.of(FeatureId.INLINE_VECTORS, FeatureId.FUSED_ADC)
        );

        // args is list of regexes, possibly needing to be split by whitespace.
        // generate a regex that matches any regex in args, or if args is empty/null, match everything
        var regex = args.length == 0 ? ".*" : Arrays.stream(args).flatMap(s -> Arrays.stream(s.split("\\s"))).map(s -> "(?:" + s + ")").collect(Collectors.joining("|"));
        // compile regex and do substring matching using find
        var pattern = Pattern.compile(regex);

        // large embeddings calculated by Neighborhood Watch.  100k files by default; 1M also available
        var coreFiles = List.of(
                "ada002-100k",
                "cohere-english-v3-100k",
                "openai-v3-small-100k",
                "nv-qa-v4-100k",
                "colbert-1M",
                "colbert-10M",
                "gecko-100k");
        executeNw(coreFiles, pattern, buildCompression, featureSets, searchCompression, mGrid, efConstructionGrid, efSearchGrid);

        var extraFiles = List.of(
                "openai-v3-large-3072-100k",
                "openai-v3-large-1536-100k",
                "e5-small-v2-100k",
                "e5-base-v2-100k",
                "e5-large-v2-100k");
        executeNw(extraFiles, pattern, buildCompression, featureSets, searchCompression, mGrid, efConstructionGrid, efSearchGrid);

        // smaller vectors from ann-benchmarks
        var hdf5Files = List.of(
                // large files not yet supported
                // "hdf5/deep-image-96-angular.hdf5",
                // "hdf5/gist-960-euclidean.hdf5",
                "glove-25-angular.hdf5",
                "glove-50-angular.hdf5",
                "lastfm-64-dot.hdf5",
                "glove-100-angular.hdf5",
                "glove-200-angular.hdf5",
                "nytimes-256-angular.hdf5",
                "sift-128-euclidean.hdf5");
        for (var f : hdf5Files) {
            if (pattern.matcher(f).find()) {
                DownloadHelper.maybeDownloadHdf5(f);
                var ds = Hdf5Loader.load(f);
                validateGroundTruth(ds);
            }
        }

        // 2D grid, built and calculated at runtime
        if (pattern.matcher("2dgrid").find()) {
            searchCompression = Arrays.asList(__ -> CompressorParameters.NONE,
                                              ds -> new PQParameters(ds.getDimension(), 256, true, UNWEIGHTED));
            buildCompression = Arrays.asList(__ -> CompressorParameters.NONE);
            var grid2d = DataSetCreator.create2DGrid(4_000_000, 10_000, 100);
            Grid.runAll(grid2d, mGrid, efConstructionGrid, featureSets, buildCompression, searchCompression, efSearchGrid);
        }
    }

    private static void executeNw(List<String> coreFiles, Pattern pattern, List<Function<DataSet, CompressorParameters>> buildCompression, List<EnumSet<FeatureId>> featureSets, List<Function<DataSet, CompressorParameters>> compressionGrid, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Double> efSearchGrid) throws IOException {
        for (var nwDatasetName : coreFiles) {
            if (pattern.matcher(nwDatasetName).find()) {
                var mfd = DownloadHelper.maybeDownloadFvecs(nwDatasetName);
                DataSet ds = mfd.load();
                validateGroundTruth(ds);
            }
        }
    }

    private static void validateGroundTruth(DataSet ds) {
        System.out.println("Validating ground truth for " + ds.name);
        int numQueriesToValidate = Math.min(10, ds.queryVectors.size());

        for (int i = 0; i < numQueriesToValidate; i++) {
            VectorFloat<?> queryVector = ds.queryVectors.get(i);
            List<SearchResult.NodeScore> groundTruth = createNodeScoresFromGroundTruth(ds, queryVector, ds.groundTruth.get(i));
            int topK = groundTruth.size();
            List<SearchResult.NodeScore> actualNearest = computeActualNearest(ds, queryVector, topK);

            // Compare scores with ground truth
            int matchingScores = compareScores(groundTruth, actualNearest);
            double accuracy = (double) matchingScores / topK;

            System.out.printf("Query %d: Score-based Accuracy = %.2f%% (%d/%d matching scores)%n", 
                              i, accuracy * 100, matchingScores, topK);
        }
    }

    private static List<SearchResult.NodeScore> createNodeScoresFromGroundTruth(DataSet ds, VectorFloat<?> queryVector, Set<Integer> groundTruthIndices) {
        return groundTruthIndices.stream()
                .map(index -> {
                    float similarity = ds.similarityFunction.compare(queryVector, ds.baseVectors.get(index));
                    return new SearchResult.NodeScore(index, similarity);
                })
                .sorted(Comparator.comparingDouble((SearchResult.NodeScore ns) -> ns.score).reversed())
                .collect(Collectors.toList());
    }

    private static List<SearchResult.NodeScore> computeActualNearest(DataSet ds, VectorFloat<?> queryVector, int topK) {
        return IntStream.range(0, ds.baseVectors.size())
                .parallel()
                .mapToObj(i -> {
                    VectorFloat<?> baseVector = ds.baseVectors.get(i);
                    float similarity = ds.similarityFunction.compare(queryVector, baseVector);
                    return new SearchResult.NodeScore(i, similarity);
                })
                .sorted(Comparator.comparingDouble((SearchResult.NodeScore ns) -> ns.score).reversed())
                .limit(topK)
                .collect(Collectors.toList());
    }

    private static int compareScores(List<SearchResult.NodeScore> groundTruth, List<SearchResult.NodeScore> actualNearest) {
        int matchingScores = 0;
        for (int i = 0; i < groundTruth.size(); i++) {
            if (Math.abs(groundTruth.get(i).score - actualNearest.get(i).score) < 1e-6) {
                matchingScores++;
            }
        }
        return matchingScores;
    }
}
