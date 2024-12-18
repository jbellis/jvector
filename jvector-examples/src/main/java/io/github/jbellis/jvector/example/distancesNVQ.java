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

import io.github.jbellis.jvector.example.util.SiftLoader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.quantization.NVQVectors;
import io.github.jbellis.jvector.quantization.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.abs;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class distancesNVQ {
    public static void testNVQEncodings(List<VectorFloat<?>> vectors, List<VectorFloat<?>> queries, VectorSimilarityFunction vsf) {
        int dimension = vectors.get(0).length();
//        int nQueries = queries.size();
//        int nVectors = vectors.size();
        int nQueries = 100;
        int nVectors = 10_000;

        vectors = vectors.subList(0, nVectors);

        // Generate a NVQ for random vectors
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var nvq = NVQuantization.compute(ravv, 2);
        nvq.learn = true;

        // Compress the vectors
        long startTime = System.nanoTime();
        var compressed = nvq.encodeAll(ravv);
        long endTime = System.nanoTime();
        double duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tEncoding took " + duration + " seconds");

        var nvqVecs = new NVQVectors(nvq, compressed);

        // compare the encoded similarities to the raw
        double distanceError = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = nvqVecs.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
                distanceError += abs(f.similarityTo(j) - vsf.compare(q, v));
            }
        }
        distanceError /= nQueries * nVectors;

        System.out.println(vsf + " error " + distanceError);
        System.out.println("--");

        float dummyAccumulator = 0;

        startTime = System.nanoTime();
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }
            var f = nvqVecs.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                dummyAccumulator += f.similarityTo(j);
            }
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tNVQ Distance computations took " + duration + " seconds");

        startTime = System.nanoTime();
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            if (VectorUtil.dotProduct(q, q) == 0) {
                continue;
            }

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
                dummyAccumulator += vsf.compare(q, v);
            }
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tFloat Distance computations took " + duration + " seconds");

        System.out.println("dummyAccumulator: " + dummyAccumulator);
    }

    public static void main(String[] args) throws IOException {
//        var baseVectors = SiftLoader.readFvecs("siftsmall/siftsmall_base.fvecs");
//        var queryVectors = SiftLoader.readFvecs("siftsmall/siftsmall_query.fvecs");

        var baseVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec");

//        var baseVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/1M/colbertv2.0_128_base_vectors_1000000.fvec");
//        var queryVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/1M/colbertv2.0_128_query_vectors_100000.fvec");

//        var baseVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec");
//        var queryVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/text-embedding-3-large_3072_100000_base_vectors.fvec");

        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());

        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.COSINE);
    }
}
