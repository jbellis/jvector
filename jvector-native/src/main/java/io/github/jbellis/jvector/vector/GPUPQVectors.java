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

import io.github.jbellis.jvector.graph.MultiAdcQuery;
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
    private final ThreadLocal<MemorySegmentVectorFloat> reusableResults;
    private final ThreadLocal<MemorySegmentByteSequence> reusableIds;

    private final int MAX_CONCURRENT_QUERIES = 256; // DEMOFIXME

    private final MemorySegment pqVectorStruct;
    private final int degree;

    private GPUPQVectors(MemorySegment pqVectorStruct, int degree) {
        this.pqVectorStruct = pqVectorStruct;
        this.degree = degree;
        this.reusableResults = ThreadLocal.withInitial(() -> new MemorySegmentVectorFloat(NativeGpuOps.allocate_float_vector(MAX_CONCURRENT_QUERIES * degree)
                                                                                                  .reinterpret(MAX_CONCURRENT_QUERIES * degree * 4))); // DEMOFIXME: use real degree
        this.reusableIds = ThreadLocal.withInitial(() -> new MemorySegmentByteSequence(NativeGpuOps.allocate_node_ids(MAX_CONCURRENT_QUERIES * degree)
                                                                                               .reinterpret(MAX_CONCURRENT_QUERIES * degree * 4)));
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
        throw new UnsupportedOperationException();
    }

    @Override
    public ScoreFunction.ApproximateScoreFunction scoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MultiAdcQuery prepareMultiAdcQuery(VectorFloat<?> queries, int queryCount) {
        assert queryCount <= MAX_CONCURRENT_QUERIES;
        var prepared = NativeGpuOps.prepare_adc_query(pqVectorStruct, ((MemorySegmentVectorFloat) queries).get(), queryCount);
        return new NativeMultiAdcQuery(prepared, reusableResults.get(), reusableIds.get());
    }

    @Override
    public int count() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    private class NativeMultiAdcQuery implements MultiAdcQuery {
        private final MemorySegment loadedQuery;
        private final MemorySegmentVectorFloat results;
        private final MemorySegmentByteSequence nodeIds;

        public NativeMultiAdcQuery(MemorySegment preparedQuery, MemorySegmentVectorFloat results, MemorySegmentByteSequence nodeIds) {
            this.loadedQuery = preparedQuery;
            this.results = results;
            this.nodeIds = nodeIds;
        }

        @Override
        public void setNodeId(int queryIdx, int offset, int nodeId) {
            nodeIds.setLittleEndianInt(queryIdx * degree + offset, nodeId);
        }

        @Override
        public int getNodeId(int queryIdx, int i) {
            return nodeIds.getLittleEndianInt(queryIdx * degree + i);
        }

        @Override
        public void computeSimilarities() {
            NativeGpuOps.compute_dp_similarities_adc(loadedQuery, nodeIds.get(), results.get(), degree);
        }

        @Override
        public float getScore(int queryIdx, int i) {
            return results.get(queryIdx * degree + i);
        }

        @Override
        public void close() {
            NativeGpuOps.free_adc_query(loadedQuery);
        }
    }
}
