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

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorEncoding;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import static java.lang.Math.min;
import static java.lang.Math.pow;

/**
 * Interface for vector compression.  T is the encoded (compressed) vector type;
 * it will be an array type.
 */
public interface VectorCompressor<T> {
    T[] encodeAll(List<float[]> vectors);

    T encode(float[] v);

    void write(DataOutput out) throws IOException;

    /**
     * @param compressedVectors must match the type T for this VectorCompressor, but
     *                          it is declared as Object because we want callers to be able to use this
     *                          without committing to a specific type T.
     */
    CompressedVectors createCompressedVectors(Object[] compressedVectors);

    static VectorCompressor<?> guessCompressorFor(GraphIndex<float[]> index, List<float[]> vectors, VectorSimilarityFunction similarityFunction) {
        var ravv = new ListRandomAccessVectorValues(vectors, vectors.get(0).length);

        // create the query vectors
        var queryVectors = IntStream.range(0, 300).mapToObj(i -> {
            return frankenvectorFrom(ravv);
        }).toArray(float[][]::new);

        // first, find the score for uncompressed queries
        int K = 100;
        var baseDelta = 0.0;
        for (var qv : queryVectors) {
            var sr = GraphSearcher.search(qv, K, ravv, VectorEncoding.FLOAT32, similarityFunction, index, Bits.ALL);
            for (int j = 0; j < min(K, sr.getNodes().length); j++) {
                baseDelta += pow(sr.getNodes()[j].score, 2);
            }
        }
        System.out.printf("score for uncompressed queries: %.2f%n", baseDelta / queryVectors.length);

        // see if we can get BQ to work
        var bq = BinaryQuantization.compute(ravv);
        var bqVectors = bq.encodeAll(vectors);
        var cv = bq.createCompressedVectors(bqVectors);
        for (int oq = 3; oq <= 5; oq++) {
            var bqDelta = 0.0;
            for (var qv : queryVectors) {
                var view = index.getView();
                NodeSimilarity.ApproximateScoreFunction sf = cv.approximateScoreFunctionFor(qv, similarityFunction);
                NodeSimilarity.ReRanker<float[]> rr = (j, vv) -> similarityFunction.compare(qv, vv.get(j));
                var sr = new GraphSearcher.Builder<>(view)
                        .build()
                        .search(sf, rr, oq * K, Bits.ALL);
                for (int j = 0; j < min(K, sr.getNodes().length); j++) {
                    bqDelta += pow(sr.getNodes()[j].score, 2);
                }
            }
            System.out.printf("score for bq oq=%s queries: %.2f%n", oq, bqDelta / queryVectors.length);
            if (bqDelta >= 0.991 * baseDelta) {
                return bq;
            }
        }

        System.out.println("no good bq found");
        return null;
    }

    // completely random vectors don't match the actual distribution well enough,
    // and using vectors from the dataset makes it too difficult to distinguish good
    // results from poor (since good results are mostly the neighbors wired into the graph),
    // so mash some together from random components that we observe in the dataset
    static float[] frankenvectorFrom(RandomAccessVectorValues<float[]> ravv) {
        var R = ThreadLocalRandom.current();
        var v = new float[ravv.dimension()];
        for (int i = 0; i < v.length; i++) {
            v[i] = ravv.vectorValue(R.nextInt(ravv.size()))[i];
        }
        return v;
    }
}
