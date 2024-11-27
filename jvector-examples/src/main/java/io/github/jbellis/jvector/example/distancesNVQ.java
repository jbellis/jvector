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
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.pq.NVQVectors;
import io.github.jbellis.jvector.pq.NVQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.min;

// this class uses explicit typing instead of `var` for easier reading when excerpted for instructional use
public class distancesNVQ {
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();

    public static void testNVQEncodings(List<VectorFloat<?>> vectors, List<VectorFloat<?>> queries, VectorSimilarityFunction vsf) {
        int dimension = vectors.get(0).length();
//        int nQueries = queries.size();
//        int nVectors = vectors.size();
        int nQueries = 100;
        int nVectors = 1_000;

        vectors = vectors.subList(0, nVectors);

        // Generate a NVQ for random vectors
        var ravv = new ListRandomAccessVectorValues(vectors, dimension);
        var nvq = NVQuantization.compute(ravv, 4, NVQuantization.BitsPerDimension.EIGHT);
//        nvq.learn = false;
//        var rec1 = nvq.encode(ravv.getVector(5)).subVectors[0].getDequantized();
        nvq.learn = true;
//        var rec2 = nvq.encode(ravv.getVector(5)).subVectors[0].getDequantized();

        // Compress the vectors
        long startTime = System.nanoTime();
        var compressed = nvq.encodeAll(ravv);
        long endTime = System.nanoTime();
        double duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tEncoding took " + duration + " seconds");

        var cv = new NVQVectors(nvq, compressed);

        // compare the encoded similarities to the raw
        startTime = System.nanoTime();
        double distanceError = 0;
        for (int i = 0; i < nQueries; i++) {
            var q = queries.get(i);
            var f = cv.scoreFunctionFor(q, vsf);

            for (int j = 0; j < nVectors; j++) {
                var v = vectors.get(j);
//                vsf.compare(q, v);

//                var d2 = vsf.compare(q, v);
//                var d1 = f.similarityTo(j);
//                System.out.println((1. / d1 - 1) + "  " + (1. / d2 - 1));
//                System.out.println(d1 + "  " + d2);

                distanceError += abs(f.similarityTo(j) - vsf.compare(q, v));
//                System.out.println(abs(f.similarityTo(j) - vsf.compare(q, v)));

            }
        }
        endTime = System.nanoTime();
        duration = (double) (endTime - startTime) / 1_000_000_000;
        System.out.println("\tDistance computations took " + duration + " seconds");

        distanceError /= nQueries * nVectors;

        System.out.println(vsf + " error " + distanceError);
        System.out.println("--");
    }

    public static void main(String[] args) throws IOException {
//        var siftPath = "siftsmall";
//        var baseVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_base.fvecs", siftPath));
//        var queryVectors = SiftLoader.readFvecs(String.format("%s/siftsmall_query.fvecs", siftPath));

        var baseVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_base_vectors.fvec");
        var queryVectors = SiftLoader.readFvecs("./fvec/wikipedia_squad/100k/ada_002_100000_query_vectors_10000.fvec");

        System.out.format("%d base and %d query vectors loaded, dimensions %d%n",
                          baseVectors.size(), queryVectors.size(), baseVectors.get(0).length());

        testNVQEncodings(baseVectors, queryVectors, VectorSimilarityFunction.DOT_PRODUCT);
    }
}
