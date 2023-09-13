package com.github.jbellis.jvector.example.util;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.github.jbellis.jvector.vector.VectorSimilarityFunction;
import com.github.jbellis.jvector.vector.VectorUtil;
import io.jhdf.HdfFile;

public class Hdf5Loader {
    public static DataSet load(String pathStr) {
        // infer the similarity
        VectorSimilarityFunction similarityFunction;
        if (pathStr.contains("angular")) {
            similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        }
        else if (pathStr.contains("euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        }
        else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + pathStr);
        }

        // read the data
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        Path path = Paths.get(pathStr);
        try (HdfFile hdf = new HdfFile(path)) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        List<float[]> scrubbedBaseVectors;
        List<float[]> scrubbedQueryVectors;
        List<Set<Integer>> gtSet;
        if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            // verify that vectors are normalized and sane
            scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
            scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
            gtSet = new ArrayList<>(groundTruth.length);
            // remove zero vectors, noting that this will change the indexes of the ground truth answers
            Map<Integer, Integer> rawToScrubbed = new HashMap<>();
            {
                int j = 0;
                for (int i = 0; i < baseVectors.length; i++) {
                    float[] v = baseVectors[i];
                    if (Math.abs(normOf(v)) > 1e-5) {
                        scrubbedBaseVectors.add(v);
                        rawToScrubbed.put(i, j++);
                    }
                }
            }
            for (int i = 0; i < queryVectors.length; i++) {
                float[] v = queryVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedQueryVectors.add(v);
                    var gt = new HashSet<Integer>();
                    for (int j = 0; j < groundTruth[i].length; j++) {
                        gt.add(rawToScrubbed.get(groundTruth[i][j]));
                    }
                    gtSet.add(gt);
                }
            }
            // now that the zero vectors are removed, we can normalize
            if (Math.abs(normOf(baseVectors[0]) - 1.0) > 1e-5) {
                normalizeAll(scrubbedBaseVectors);
                normalizeAll(scrubbedQueryVectors);
            }
            assert scrubbedQueryVectors.size() == gtSet.size();
        }
        else {
            scrubbedBaseVectors = Arrays.asList(baseVectors);
            scrubbedQueryVectors = Arrays.asList(queryVectors);
            gtSet = new ArrayList<>(groundTruth.length);
            for (int[] gt : groundTruth) {
                var gtSetForQuery = new HashSet<Integer>();
                for (int i : gt) {
                    gtSetForQuery.add(i);
                }
                gtSet.add(gtSetForQuery);
            }
        }

        System.out.format("%n%s: %d base and %d query vectors loaded, dimensions %d%n",
                          pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        return new DataSet(path.getFileName().toString(), similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }
}
