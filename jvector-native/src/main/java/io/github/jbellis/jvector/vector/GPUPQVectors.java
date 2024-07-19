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

package io.github.jbellis.jvector.vector;

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.pq.CompressedVectors;
import io.github.jbellis.jvector.pq.VectorCompressor;
import io.github.jbellis.jvector.vector.cnative.NativeGpuOps;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

public class GPUPQVectors implements CompressedVectors {
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ThreadLocal<VectorFloat<?>> reusableResults;

    private final MemorySegment pqVectorStruct;
    private final int degree;

    private GPUPQVectors(MemorySegment pqVectorStruct, int degree) {
        this.pqVectorStruct = pqVectorStruct;
        this.degree = degree;
        this.reusableResults = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(128)); // DEMOFIXME: use real degree
    }

    public static GPUPQVectors load(Path pqVectorsPath, int degree) {
        MemorySegment pqVectorStruct = NativeGpuOps.load_pq_vectors(MemorySegment.ofArray(pqVectorsPath.toString().getBytes()));
        return new GPUPQVectors(pqVectorStruct, degree);
    }

    @Override
    public void write(DataOutput out, int version) throws IOException {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public int getOriginalSize() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public int getCompressedSize() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public VectorCompressor<?> getCompressor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction precomputedScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return scoreFunctionFor(q, similarityFunction);
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        MemorySegment loadedQuery = NativeGpuOps.prepare_adc_query(pqVectorStruct, ((MemorySegmentVectorFloat) q).get(), degree);
        return new GPUApproximateScoreFunction() {
            private final VectorFloat<?> results = reusableResults.get();
            @Override
            public float similarityTo(int node2) {
                return similarityTo(new int[]{node2}).get(0);
            }

            @Override
            public boolean supportsMultinodeSimilarity() {
                return true;
            }

            @Override
            public VectorFloat<?> similarityTo(int[] nodeIds) {
                NativeGpuOps.compute_dp_similarities_adc(loadedQuery, MemorySegment.ofArray(nodeIds), ((MemorySegmentVectorFloat) results).get(), nodeIds.length);
                return results;
            }

            @Override
            public void close() {
                NativeGpuOps.free_adc_query(loadedQuery);
            }
        };
    }

    @Override
    public int count() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    // DEMOFIXME is there a better way to expose this?
    public interface GPUApproximateScoreFunction extends ScoreFunction.ApproximateScoreFunction {
        void close();
    }
}
