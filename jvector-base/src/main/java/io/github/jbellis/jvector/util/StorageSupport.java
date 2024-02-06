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

import java.util.function.Supplier;

/**
 * Abstracts whether a stored value is thread-local or shared across threads.
 *
 * @param <T> The object to be stored by this instance.
 */
public abstract class StorageSupport<T> {

    /**
     * Creates thread-local storage for the given type.
     * This is a replacement for ThreadLocal, used for callsites where the value may need to be
     * unique per-thread or shared across threads.
     * @param initialValue allows creation of new instances for the pool
     */
    public static <T> StorageSupport<T> newThreadLocal(Supplier<T> initialValue) {
        return new ThreadLocalStorage<>(initialValue);
    }

    /**
     * Special case of not needing thread-local storage. Same fixed value is returned for all threads.
     *
     * @param fixedValue the value this pool will always return
     */
    public static <T> StorageSupport<T> newShared(T fixedValue) {
        return new SharedStorage<>(fixedValue);
    }

    /**
     * @return a value of type T, either the same for all threads or unique to each thread.
     */
    public abstract T get();

    private StorageSupport() {
    }

    final static class ThreadLocalStorage<T> extends StorageSupport<T>
    {
        private final ThreadLocal<T> threadLocal;

        private ThreadLocalStorage(Supplier<T> initialValue) {
            this.threadLocal = ThreadLocal.withInitial(initialValue);
        }
;
        @Override
        public T get() {
            return threadLocal.get();
        }
    }


    final static class SharedStorage<T> extends StorageSupport<T> {
        private final T value;
        private SharedStorage(T value) {
            this.value = value;
        }

        @Override
        public T get() {
            return value;
        }
    }
}
