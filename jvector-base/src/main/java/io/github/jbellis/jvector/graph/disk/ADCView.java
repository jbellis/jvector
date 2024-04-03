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

package io.github.jbellis.jvector.graph.disk;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public interface ADCView extends GraphIndex.RerankingView, RandomAccessVectorValues {
    /**
     * @return the quantized, transposed, and packed vectors of the given node's neighbors.
     * Only one ByteSequence is allocated by the View, so if you need to keep multiple
     * results, you should copy it manually.
     */
    ByteSequence<?> getPackedNeighbors(int node);

    /**
     * @return a vector for storing ADC result into, backed by a per-Graph threadlocal.
     */
    VectorFloat<?> reusableResults();

    ProductQuantization getProductQuantization();

    ScoreFunction.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction);
}
