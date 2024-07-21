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

import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
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
    private static final VectorTypeSupport vts = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ThreadLocal<MemorySegmentVectorFloat> reusableResults;
    private final ThreadLocal<MemorySegmentVectorFloat> reusableVectors;

    private final MemorySegment pqVectorStruct;
    private final RandomAccessVectorValues ravv;
    private final int batchSize;

    private GPUPQVectors(MemorySegment pqVectorStruct, int batchSize, int dimension) {
        this(pqVectorStruct, null, batchSize, dimension);
    }

    public GPUPQVectors(RandomAccessVectorValues ravv, int batchSize) {
        this(null, ravv, batchSize, ravv.dimension());
    }

    private GPUPQVectors(MemorySegment pqVectorStruct, RandomAccessVectorValues ravv, int batchSize, int dimension) {
        this.pqVectorStruct = pqVectorStruct;
        this.ravv = ravv;
        this.batchSize = batchSize;
        this.reusableResults = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.cuda_allocate(batchSize * Float.BYTES)
                                                                                                  .reinterpret(batchSize * Float.BYTES)));
        this.reusableVectors = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.cuda_allocate(batchSize * dimension * Float.BYTES)
                                                                                                   .reinterpret(batchSize * dimension * Float.BYTES)));
    }

    public static GPUPQVectors load(Path pqVectorsPath, int batchSize, int dimension) {
        MemorySegment pqVectorStruct = NativeGpuOps.load_pq_vectors(MemorySegment.ofArray(pqVectorsPath.toString().getBytes()));
        return new GPUPQVectors(pqVectorStruct, batchSize, dimension); // DEMOFIXME
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
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        throw new UnsupportedOperationException();
    }

    public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        return new GPUExactScoreFunction() {
            private final MemorySegmentVectorFloat results = reusableResults.get();
            private final MemorySegmentVectorFloat vectors = reusableVectors.get(); // DEMOFIXME HACK

            /** slow!  do not use!  implemented for tests */
            @Override
            public float similarityTo(int node2) {
                vectors.copyFrom(ravv.getVector(node2), 0, 0, q.length());
                NativeGpuOps.compute_dp_similarities_raw(((MemorySegmentVectorFloat) q).get(), vectors.get(), q.length(), results.get(), 1);
                return results.get(0);
            }

            @Override
            public boolean supportsMultinodeSimilarity() {
                return true;
            }

            @Override
            public VectorFloat<?> similarityTo(NodesIterator nodeIds) {
                int length = nodeIds.size();
                for (int i = 0; i < length; i++) {
                    var id = nodeIds.next();
                    vectors.copyFrom(ravv.getVector(id), 0, i * q.length(), q.length());
                }
                NativeGpuOps.compute_dp_similarities_raw(((MemorySegmentVectorFloat) q).get(), vectors.get(), q.length(), results.get(), length);
                return results;
            }

            @Override
            public void close() {
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

    // DEMOFIXME is there a better way to expose this?
    public interface GPUExactScoreFunction extends ScoreFunction.ExactScoreFunction {
        void close();
    }
}
