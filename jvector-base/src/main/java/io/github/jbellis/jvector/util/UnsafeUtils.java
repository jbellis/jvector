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
package io.github.jbellis.jvector.util;

import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Locale;

public class UnsafeUtils {

    private static final Unsafe UNSAFE = getUnsafe();
    private static final Field ADDRESS_FIELD = getBufferAddressField();
    private static final long UNSAFE_COPY_THRESHOLD = 1024L * 1024L * 4;
    private static final long BYTE_ARRAY_BASE_OFFSET = UNSAFE.arrayBaseOffset(byte[].class);


    private static Unsafe getUnsafe() {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            return (Unsafe) f.get(null);
        } catch (Exception e) {
            throw new IllegalStateException(UnsafeUtils.class.getSimpleName() + " can't acquire needed Unsafe access");
        }
    }

    private static Field getBufferAddressField() {
        try {
            var field = Buffer.class.getDeclaredField("address");
            field.setAccessible(true);
            return field;
        } catch (NoSuchFieldException e) {
            throw new IllegalStateException("Init failed, no 'address' in Buffer");
        }
    }

    public static long getDirectBufferAddress(ByteBuffer byteBuffer) {
        try {
            return (long) ADDRESS_FIELD.get(byteBuffer);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("Get address of byteBuffer failed");
        }
    }

    public static void getBytes(long srcAddr, byte[] dst, long dstOffset, long length) {
        if ((dstOffset | length | (dstOffset + length) | (dst.length - (dstOffset + length))) < 0) {
            throw new IndexOutOfBoundsException(String.format(Locale.ROOT, "arraySize:%d, offset:%d, len:%d", dst.length, dstOffset, length));
        }
        long offset = BYTE_ARRAY_BASE_OFFSET + dstOffset;
        while (length > 0) {
            long size = Math.min(length, UNSAFE_COPY_THRESHOLD);
            UNSAFE.copyMemory(null, srcAddr, dst, offset, size);
            length -= size;
            srcAddr += size;
            offset += size;
        }
    }

    public static void copyBytes(byte[] src, long srcOffset, long length, long targetAddr) {
        if ((srcOffset | length | (src.length - (srcOffset + length))) < 0) {
            throw new IndexOutOfBoundsException(String.format(Locale.ROOT, "arraySize:%d, offset:%d, len:%d", src.length, srcOffset, length));
        }
        long offset = BYTE_ARRAY_BASE_OFFSET + srcOffset;
        UNSAFE.copyMemory(src, offset, null, targetAddr, length);
    }
}
