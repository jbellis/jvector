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
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * VectorTypeSupport implementation that uses on-heap arrays for ByteSequence/VectorFloat.
 */
final class ArrayVectorProvider implements VectorTypeSupport
{

    @Override
    public VectorFloat<?> createFloatVector(Object data)
    {
        return new ArrayVectorFloat((float[]) data);
    }

    @Override
    public VectorFloat<?> createFloatVector(int length)
    {
        return new ArrayVectorFloat(length);
    }

    @Override
    public VectorFloat<?> readFloatVector(RandomAccessReader r, int size) throws IOException
    {
        float[] vector = new float[size];
        r.readFully(vector);
        return new ArrayVectorFloat(vector);
    }

    @Override
    public void readFloatVector(RandomAccessReader r, int size, VectorFloat<?> vector, int offset) throws IOException {
        r.read(((ArrayVectorFloat) vector).get(), offset, size);
    }

    @Override
    public void writeFloatVector(DataOutput out, VectorFloat<?> vector) throws IOException
    {
        ArrayVectorFloat v = (ArrayVectorFloat)vector;
        // this seems to be the only way to avoid writing float-at-a-time which is far too slow
        var fb = FloatBuffer.wrap(v.get());
        var bb = ByteBuffer.allocate(fb.capacity() * Float.BYTES);
        bb.asFloatBuffer().put(fb);
        bb.rewind();
        out.write(bb.array());
    }

    @Override
    public ByteSequence<?> createByteSequence(Object data)
    {
        return new ArrayByteSequence((byte[]) data);
    }

    @Override
    public ByteSequence<?> createByteSequence(int length)
    {
        return new ArrayByteSequence(length);
    }

    @Override
    public byte[] readBytes(RandomAccessReader r, int size) throws IOException
    {
        byte[] vector = new byte[size];
        r.readFully(vector);
        return vector;
    }

    @Override
    public ByteSequence<?> readByteSequence(RandomAccessReader r, int size) throws IOException
    {
        return new ArrayByteSequence(readBytes(r, size));
    }

    @Override
    public void readByteSequence(RandomAccessReader r, ByteSequence<?> sequence) throws IOException {
        ArrayByteSequence v = (ArrayByteSequence) sequence;
        r.readFully(v.get());
    }

    @Override
    public void writeByteSequence(DataOutput out, ByteSequence<?> sequence) throws IOException
    {
        ArrayByteSequence v = (ArrayByteSequence) sequence;
        out.write(v.get());
    }

    @Override
    public void writeBytes(DataOutput out, Object sequence) throws IOException
    {
        out.write((byte[]) sequence);
    }
}
