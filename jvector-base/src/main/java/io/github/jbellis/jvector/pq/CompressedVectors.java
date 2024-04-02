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

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.DataOutput;
import java.io.IOException;

public interface CompressedVectors extends Accountable {
    /** write the compressed vectors to the given DataOutput */
    void write(DataOutput out) throws IOException;

    /** @return the original size of each vector, in bytes, before compression */
    int getOriginalSize();

    /** @return the compressed size of each vector, in bytes */
    int getCompressedSize();

    /** @return the compressor used by this instance */
    VectorCompressor<?> getCompressor();

    /** precomputes partial scores for the given query with every centroid; suitable for most searches */
    ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    /** no precomputation; suitable when just a handful of score computations are performed */
    ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);

    @Deprecated
    default ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return precomputedScoreFunctionFor(q, similarityFunction);
    }

    /** the number of vectors */
    int count();
}
