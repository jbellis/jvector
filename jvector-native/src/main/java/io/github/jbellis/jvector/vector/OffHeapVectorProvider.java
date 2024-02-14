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

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.nio.Buffer;

/**
 * VectorTypeSupport using off-heap MemorySegments.
 */
public class OffHeapVectorProvider implements VectorTypeSupport
{
    @Override
    public VectorFloat<?> createFloatVector(Object data)
    {
        if (data instanceof Buffer)
            return new OffHeapVectorFloat((Buffer) data);

        return new OffHeapVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatVector(int length)
    {
        return new OffHeapVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException
    {
        float[] d = new float[size];
        r.readFully(d);
        return new OffHeapVectorFloat(d);
    }

    @Override
    public void writeFloatVector(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        for (int i = 0; i < vector.length(); i++)
            out.writeFloat(vector.get(i));
    }

    @Override
    public ByteSequence<?> createByteSequence(Object data)
    {
        if (data instanceof Buffer)
            return new OffHeapByteSequence((Buffer) data);

        return new OffHeapByteSequence((byte[]) data);
    }

    @Override
    public ByteSequence<?> createByteSequence(int length)
    {
        return new OffHeapByteSequence(length);
    }

    @Override
    public ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException
    {
        var vector = new OffHeapByteSequence(size);
        r.readFully(vector.get().asByteBuffer());
        return vector;
    }

    @Override
    public void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException {
        r.readFully(((OffHeapByteSequence) sequence).get().asByteBuffer());
    }


    @Override
    public void writeByteSequence(DataOutput out, ByteSequence<?> sequence) throws IOException
    {
        for (int i = 0; i < sequence.length(); i++)
            out.writeByte(sequence.get(i));
    }
}
