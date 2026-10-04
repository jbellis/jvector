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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class LazyMemorySegmentTest {
    @Test
    void concurrentFirstUsePublishesOneViewOfTheOriginalArray() throws Exception {
        float[] data = {-0.0f, 1, 2, 3, 4, 5, 6, 7};
        var v = new MemorySegmentVectorFloat(data);
        assertSame(data, v.array());
        var field = MemorySegmentVectorFloat.class.getDeclaredField("segment");
        field.setAccessible(true);
        assertNull(field.get(v));
        try (var pool = Executors.newFixedThreadPool(48)) {
            var barrier = new CyclicBarrier(48);
            var tasks = new ArrayList<Future<MemorySegment>>();
            for (int i = 0; i < 48; i++) tasks.add(pool.submit(() -> {
                barrier.await();
                var first = v.get();
                for (int n = 0; n < 10000; n++) assertSame(first, v.get());
                return first;
            }));
            MemorySegment first = tasks.get(0).get();
            for (var task : tasks) assertSame(first, task.get());
            v.set(3, 19);
            assertEquals(19, first.getAtIndex(ValueLayout.JAVA_FLOAT, 3));
            first.setAtIndex(ValueLayout.JAVA_FLOAT, 4, 23);
            assertEquals(23, v.get(4));
        }
    }

    @Test
    void copiesZeroBufferAndNativeUseKeepStorageSemantics() throws Exception {
        float[] data = {-0.0f, Float.intBitsToFloat(0x7fc00123), 2, 3, 4, 5, 6, 7};
        var original = new MemorySegmentVectorFloat(data);
        var copy = (MemorySegmentVectorFloat) original.copy();
        var field = MemorySegmentVectorFloat.class.getDeclaredField("segment");
        field.setAccessible(true);
        assertNull(field.get(original));
        assertNull(field.get(copy));
        assertNotSame(data, copy.array());
        for (int i = 0; i < data.length; i++)
            assertEquals(Float.floatToRawIntBits(data[i]), Float.floatToRawIntBits(copy.get(i)));
        assertEquals(original, copy);
        copy.copyFrom(copy, 0, 1, 7);
        assertEquals(0x80000000, Float.floatToRawIntBits(copy.get(1)));
        assertEquals(6, copy.get(7));
        var buffer = FloatBuffer.wrap(new float[]{9, 10, 11});
        buffer.position(1);
        var fromBuffer = new MemorySegmentVectorFloat(buffer);
        assertArrayEquals(new float[]{10, 11}, fromBuffer.array());
        assertEquals(2, fromBuffer.length());
        var support = new NativeVectorizationProvider().getVectorUtilSupport();
        assertEquals(0, support.squareDistance(fromBuffer, fromBuffer));
        fromBuffer.zero();
        for (float value : fromBuffer.array()) assertEquals(0, Float.floatToRawIntBits(value));
    }
}
