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

package io.github.jbellis.jvector.disk;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;

/**
 * {@link MappingAdvice} through {@code posix_madvise}, the same downcall {@link MemorySegmentReader}
 * uses for {@code MADV_RANDOM} on the source files. Loaded reflectively by {@link MappingAdvice#get()};
 * the constructor fails when the symbol is missing so the caller falls back.
 */
public final class NativeMappingAdvice implements MappingAdvice {
    private static final int POSIX_MADV_WILLNEED = 3;
    private static final long PAGE = 4096;
    private final MethodHandle madvise;

    public NativeMappingAdvice() {
        var linker = Linker.nativeLinker();
        var symbol = linker.defaultLookup().find("posix_madvise")
                .orElseThrow(() -> new UnsupportedOperationException("posix_madvise not found"));
        madvise = linker.downcallHandle(symbol,
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
    }

    @Override
    public void willNeed(ByteBuffer buffer, int offset, int length) {
        if (length <= 0 || !buffer.isDirect()) {
            return;
        }
        MemorySegment whole = MemorySegment.ofBuffer(buffer);
        long start = whole.address() + offset;
        long aligned = start & -PAGE;                       // madvise wants a page-aligned start
        long end = Math.min(start + length, whole.address() + whole.byteSize());
        MemorySegment span = MemorySegment.ofAddress(aligned).reinterpret(end - aligned);
        try {
            int ignored = (int) madvise.invokeExact(span, span.byteSize(), POSIX_MADV_WILLNEED);
        } catch (Throwable t) {
            // advice is best effort
        }
    }
}
