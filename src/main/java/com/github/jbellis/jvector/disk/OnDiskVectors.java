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

import com.github.jbellis.jvector.graph.RandomAccessVectorValues;

import java.io.IOException;
import java.io.UncheckedIOException;

public class OnDiskVectors implements RandomAccessVectorValues<float[]>, AutoCloseable
{
    private final int dimension;
    private final int size;
    private final float[] vector;
    private final RandomAccessReader in;
    private final long segmentOffset;

    public OnDiskVectors(RandomAccessReader in, long offset)
    {
        this.in = in;
        this.segmentOffset = offset;
        try {
            this.size = in.readInt();
            this.dimension = in.readInt();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        assert dimension < 100000 : dimension; // sanity check
        this.vector = new float[dimension];
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public int dimension()
    {
        return dimension;
    }

    @Override
    public float[] vectorValue(int i)
    {
        try
        {
            readVector(i, vector);
            return vector;
        }
        catch (IOException e)
        {
            throw new UncheckedIOException(e);
        }
    }

    void readVector(int i, float[] v) throws IOException
    {
        in.readFloatsAt(segmentOffset + 8L + i * dimension * 4L, v);
    }

    @Override
    public RandomAccessVectorValues<float[]> copy()
    {
        // TODO maybe we actually need this?  in which case we'll need RAR to provide a copy() method too
        // this is only necessary if we need to build a new graph from this vector source.
        // (the idea is that if you are re-using float[] between calls, like we are here,
        //  you can make a copy of the source so you can compare different neighbors' scores
        //  as you build the graph.)
        // since we only build new graphs during insert and compaction, when we get the vectors from the source rows,
        // we don't need to worry about this.
        throw new UnsupportedOperationException();
    }

    @Override
    public void close() throws Exception {
        in.close();
    }
}
