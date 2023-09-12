package com.github.jbellis.jvector.example.util;

import java.util.List;
import java.util.Set;

import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

public class DataSet {
    public final String name;
    public final VectorSimilarityFunction similarityFunction;
    public final List<float[]> baseVectors;
    public final List<float[]> queryVectors;
    public final List<? extends Set<Integer>> groundTruth;

    public DataSet(String name, VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<? extends Set<Integer>> groundTruth) {
        this.name = name;
        this.similarityFunction = similarityFunction;
        this.baseVectors = baseVectors;
        this.queryVectors = queryVectors;
        this.groundTruth = groundTruth;
    }
}
