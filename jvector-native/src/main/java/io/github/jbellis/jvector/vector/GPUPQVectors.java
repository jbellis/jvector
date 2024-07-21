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
    private final ThreadLocal<MemorySegmentByteSequence> reusableIds;

    private final MemorySegment pqVectorStruct;
    private final int degree;

    private GPUPQVectors(MemorySegment pqVectorStruct, int degree) {
        this.pqVectorStruct = pqVectorStruct;
        this.degree = degree;
        this.reusableResults = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.allocate_results(degree).reinterpret(degree * 4))); // DEMOFIXME: use real degree
        this.reusableIds = ThreadLocal.withInitial(() -> new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(degree).reinterpret(degree * 4)));
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
        MemorySegment loadedQuery = NativeGpuOps.prepare_query(pqVectorStruct, ((MemorySegmentVectorFloat) q).get());
        return new GPUApproximateScoreFunction() {
            private final MemorySegmentVectorFloat results = reusableResults.get();
            private final MemorySegmentByteSequence ids = reusableIds.get(); // DEMOFIXME HACK

            @Override
            public float similarityTo(int node2) {
                ids.setLittleEndianInt(0, node2);
                NativeGpuOps.compute_dp_similarities(loadedQuery, ids.get(), results.get(), 1);
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
                    ids.setLittleEndianInt(i, nodeIds.nextInt());
                }
                NativeGpuOps.compute_dp_similarities(loadedQuery, ids.get(), results.get(), length);
                return results;
            }

            @Override
            public void close() {
                NativeGpuOps.free_query(loadedQuery);
            }
        };
    }

    public ScoreFunction.ExactScoreFunction rerankerFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        var sf = (GPUApproximateScoreFunction) scoreFunctionFor(q, similarityFunction);
        return new GPUExactScoreFunction() {
            @Override
            public float similarityTo(int node2) {
                return sf.similarityTo(node2);
            }

            @Override
            public void close() {
                sf.close();
            }

            @Override
            public VectorFloat<?> similarityTo(NodesIterator nodeIds) {
                return sf.similarityTo(nodeIds);
            }

            @Override
            public boolean supportsMultinodeSimilarity() {
                return true;
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
