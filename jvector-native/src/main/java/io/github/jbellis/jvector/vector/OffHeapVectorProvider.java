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
import java.nio.ByteOrder;

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
        var vector = new OffHeapVectorFloat(size);
        var buffer = vector.get().asByteBuffer();
        r.readFully(buffer);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.flip();
        for (int i = 0; i < buffer.capacity(); i = i + 4) {
            var val = buffer.getInt(i);
            buffer.putInt(i, Integer.reverseBytes(val));
        }
        return vector;
    }

    @Override
    public void readFloatVector(RandomAccessReader r, int count, VectorFloat<?> vector, int offset) throws IOException {
        var destBuffer = ((OffHeapVectorFloat) vector).get().asByteBuffer();
        destBuffer.position(offset * Float.BYTES);
        destBuffer.limit(destBuffer.position() + count * Float.BYTES);
        r.readFully(destBuffer);
        destBuffer.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = offset * Float.BYTES; i < destBuffer.limit(); i = i + 4) {
            var val = destBuffer.getInt(i);
            destBuffer.putInt(i, Integer.reverseBytes(val));
        }
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
