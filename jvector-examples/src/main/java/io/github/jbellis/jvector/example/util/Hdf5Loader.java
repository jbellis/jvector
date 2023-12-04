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

package io.github.jbellis.jvector.example.util;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.jhdf.HdfFile;
import io.jhdf.api.Dataset;
import io.jhdf.object.datatype.FloatingPoint;

import java.nio.file.Path;
import java.util.*;
import java.util.stream.IntStream;

public class Hdf5Loader {
    public static final String HDF5_DIR = "hdf5/";

    public static DataSet load(String filename) {
        // infer the similarity
        VectorSimilarityFunction similarityFunction;
        if (filename.contains("-angular") || filename.contains("-dot")) {
            similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        }
        else if (filename.contains("-euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        }
        else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + filename);
        }

        // read the data
        float[][] baseVectors;
        float[][] queryVectors;
        Path path = Path.of(HDF5_DIR).resolve(filename);
        var gtSets = new ArrayList<HashSet<Integer>>();
        try (HdfFile hdf = new HdfFile(path)) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            Dataset queryDataset = hdf.getDatasetByPath("test");
            if (((FloatingPoint) queryDataset.getDataType()).getBitPrecision() == 64) {
                // lastfm dataset contains f64 queries but f32 everything else
                var doubles = ((double[][]) queryDataset.getData());
                queryVectors = IntStream.range(0, doubles.length).parallel().mapToObj(i -> {
                    var a = new float[doubles[i].length];
                    for (int j = 0; j < doubles[i].length; j++) {
                        a[j] = (float) doubles[i][j];
                    }
                    return a;
                }).toArray(float[][]::new);
            } else {
                queryVectors = (float[][]) queryDataset.getData();
            }
            int[][] groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
            gtSets = new ArrayList<>(groundTruth.length);
            for (int[] i : groundTruth) {
                var gtSet = new HashSet<Integer>(i.length);
                for (int j : i) {
                    gtSet.add(j);
                }
                gtSets.add(gtSet);
            }
        }

        return DataSet.getScrubbedDataSet(path.getFileName().toString(), similarityFunction, Arrays.asList(baseVectors), Arrays.asList(queryVectors), gtSets);
    }
}
