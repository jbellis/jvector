/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.jbellis.jvector.disk;

import com.github.jbellis.jvector.pq.ProductQuantization;
import com.github.jbellis.jvector.vector.VectorSimilarityFunction;

import java.io.DataInput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CompressedVectors
{
    private final ProductQuantization pq;
    private final List<byte[]> compressedVectors;

    public CompressedVectors(ProductQuantization pq, List<byte[]> compressedVectors)
    {
        this.pq = pq;
        this.compressedVectors = compressedVectors;
    }

    public static CompressedVectors load(RandomAccessReader in, long offset) throws IOException
    {
        in.seek(offset);
        if (!in.readBoolean()) {
            // there were too few vectors to bother compressing
            return null;
        }

        // pq codebooks
        var pq = ProductQuantization.load(in);

        // read the vectors
        int size = in.readInt();
        var compressedVectors = new ArrayList<byte[]>(size);
        int compressedDimension = in.readInt();
        for (int i = 0; i < size; i++)
        {
            byte[] vector = new byte[compressedDimension];
            in.readFully(vector);
            compressedVectors.add(vector);
        }

        return new CompressedVectors(pq, compressedVectors);
    }

    public float decodedSimilarity(int ordinal, float[] v, VectorSimilarityFunction similarityFunction)
    {
        switch (similarityFunction)
        {
            case DOT_PRODUCT:
                return (1 + pq.decodedDotProduct(compressedVectors.get(ordinal), v)) / 2;
            default:
                // TODO implement other similarity functions efficiently
                var decoded = new float[pq.vectorDimension()];
                pq.decode(compressedVectors.get(ordinal), decoded);
                return similarityFunction.compare(decoded, v);
        }
    }
}
