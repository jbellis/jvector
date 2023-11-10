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

import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

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
        var R = ThreadLocalRandom.current();
        var ravv = new ListRandomAccessVectorValues(vectors, vectors.get(0).length);

        // first, find the 1@K recall for uncompressed queries
        int K = 4;
        var baseRecall = 0.0;
        int N_QUERIES = 200;
        for (int i = 0; i < N_QUERIES; i++) {
            int qOrd = R.nextInt(ravv.size());
            var qv = ravv.vectorValue(qOrd);
            var sr = GraphSearcher.search(qv, K, ravv, VectorEncoding.FLOAT32, similarityFunction, index, Bits.ALL);
            baseRecall += sr.getNodes()[0].node == qOrd ? 1 : 0;
        }
        System.out.printf("recall for uncompressed queries: %.2f%n", baseRecall / N_QUERIES);

        // see if we can get BQ to work
        var bq = BinaryQuantization.compute(ravv);
        var bqVectors = bq.encodeAll(vectors);
        var cv = bq.createCompressedVectors(bqVectors);
        for (int oq = 3; oq <= 5; oq++) {
            var bqRecall = 0.0;
            for (int i = 0; i < N_QUERIES; i++) {
                int qOrd = R.nextInt(ravv.size());
                var qv = ravv.vectorValue(qOrd);

                var view = index.getView();
                NodeSimilarity.ApproximateScoreFunction sf = cv.approximateScoreFunctionFor(qv, similarityFunction);
                NodeSimilarity.ReRanker<float[]> rr = (j, vv) -> similarityFunction.compare(qv, vv.get(j));
                var sr = new GraphSearcher.Builder<>(view)
                        .build()
                        .search(sf, rr, oq * K, Bits.ALL);
                bqRecall += sr.getNodes()[0].node == qOrd ? 1 : 0;
            }
            System.out.printf("recall for bq oq=%s queries: %.2f%n", oq, bqRecall / N_QUERIES);
            if (bqRecall >= 0.99 * baseRecall) {
                return bq;
            }
        }

        System.out.println("no good bq found");
        return null;
    }
}
