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

import java.nio.ByteBuffer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Page-cache advice for a memory-mapped {@link ByteBuffer}. The JDK gives a mapping no way to
 * talk to {@code madvise}; the native module (JDK 22+) supplies an implementation through the
 * foreign function API, and {@link #get()} finds it the way {@link ReaderSupplierFactory} finds
 * the native reader. Callers treat a missing implementation as "no advice available" and fall
 * back to plain reads.
 */
public interface MappingAdvice {
    String NATIVE_CLASSNAME = "io.github.jbellis.jvector.disk.NativeMappingAdvice";

    /**
     * Asks the kernel to read {@code length} bytes of the mapping {@code buffer} starting at
     * {@code offset} into the page cache asynchronously ({@code POSIX_MADV_WILLNEED}). Pages
     * already resident cost nothing. Best effort: failures are ignored.
     */
    void willNeed(ByteBuffer buffer, int offset, int length);

    /** The native implementation, or {@code null} when it cannot be loaded. */
    static MappingAdvice get() {
        return Holder.INSTANCE;
    }

    /** Lazy holder so the lookup happens once, on first use. */
    final class Holder {
        private static final Logger LOG = Logger.getLogger(MappingAdvice.class.getName());
        static final MappingAdvice INSTANCE = load();

        private Holder() {
        }

        private static MappingAdvice load() {
            try {
                var clazz = Class.forName(NATIVE_CLASSNAME);
                return (MappingAdvice) clazz.getConstructor().newInstance();
            } catch (Throwable e) {
                LOG.log(Level.FINE, "Native mapping advice not available; mapped stores are streamed with positional reads", e);
                return null;
            }
        }
    }
}
