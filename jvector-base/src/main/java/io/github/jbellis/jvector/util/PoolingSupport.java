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

import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Allows any object to be pooled and released when work is done.
 * This is an alternative to using {@link ThreadLocal}.
 *
 * @param <T> The object to be pooled by this instance
 */
public abstract class PoolingSupport<T> {

    /**
     * Creates a pool of objects intended to be used by a thread pool.
     * This is a replacement for ThreadLocal.
     * @param initialValue allows creation of new instances for the pool
     */
    public static <T> PoolingSupport<T> newThreadBased(Supplier<T> initialValue) {
        return new ThreadPooling<>(initialValue);
    }

    /**
     * Special case of not actually needing a pool (when other times you do)
     *
     * @param fixedValue the value this pool will always return
     */
    public static <T> PoolingSupport<T> newNoPooling(T fixedValue) {
        return new NoPooling<>(fixedValue);
    }


    /**
     * Recycling of objects using a MPMC queue
     *
     * @param limit the specific number of threads to be sharing the pooled objects
     * @param initialValue allows creation of new instances for the pool
     */
    public static <T> PoolingSupport<T> newQueuePooling(int limit, Supplier<T> initialValue) {
        return new QueuedPooling<>(limit, initialValue);
    }

    private PoolingSupport() {
    }

    /**
     * @return a pooled object which will be returned to the pool when thread is finished with it
     */
    public abstract Pooled<T> get();

    /**
     * This call returns all values what are in the pool, for the case of work spread across many pooled objects
     * then processed after they are finished.
     *
     * @return a stream of everything in the pool.
     * @throws IllegalStateException if outstanding items are not yet returned to the pool
     */
    public abstract Stream<T> stream();

    /**
     * Internal call used when pooled item is returned
     * @param value
     */
    protected abstract void onClosed(Pooled<T> value);

    /**
     * Wrapper class for items in the pool
     *
     * These are AutoClosable and are intended to be used
     * in a try-with-resources statement.
     * @param <T>
     */
    public final static class Pooled<T> implements AutoCloseable {
        private final T value;
        private final PoolingSupport<T> owner;
        private Pooled(PoolingSupport<T> owner, T value) {
            this.owner = owner;
            this.value = value;
        }

        public T get() {
            return value;
        }

        @Override
        public void close() {
            if (owner != null)
                owner.onClosed(this);
        }
    }


    final static class ThreadPooling<T> extends PoolingSupport<T>
    {
        private final ThreadLocal<Pooled<T>> threadLocal;
        private final Supplier<T> initialValue;

        private ThreadPooling(Supplier<T> initialValue) {
            this.initialValue = initialValue;
            this.threadLocal = new ThreadLocal<>();
        }

        @Override
        public Pooled<T> get() {
            Pooled<T> val = threadLocal.get();
            if (val != null)
                return val;

            // We pass null as the owner to prevent a memory leak. If we passed 'this' as the owner and then put val
            // into the threadLocal, then even after 'this' is inaccessable from the application code, it would still
            // have a strong reference in the ThreadLocalMap, thus preventing this object from being garbage collected
            // and preventing the value in threadLocal from being garbage collected.
            val = new Pooled<>(null, initialValue.get());
            threadLocal.set(val);
            return val;
        }

        @Override
        public Stream<T> stream() {
            throw new UnsupportedOperationException();
        }

        @Override
        protected void onClosed(Pooled<T> value) {

        }
    }


    final static class NoPooling<T> extends PoolingSupport<T> {
        private final T value;
        private final Pooled<T> staticPooled;
        private NoPooling(T value) {
            this.value = value;
            this.staticPooled = new Pooled<>(this, this.value);
        }

        @Override
        public Pooled<T> get() {
            return staticPooled;
        }

        @Override
        public Stream<T> stream() {
            return Stream.of(value);
        }

        @Override
        protected void onClosed(Pooled<T> value) {
        }
    }


    final static class QueuedPooling<T> extends PoolingSupport<T> {
        private final int limit;
        private final AtomicInteger created;
        private final LinkedBlockingQueue<Pooled<T>> queue;
        private final Supplier<T> initialValue;

        private QueuedPooling(int limit, Supplier<T> initialValue) {
            this.limit = limit;
            this.created = new AtomicInteger(0);
            this.queue = new LinkedBlockingQueue<>(limit);
            this.initialValue = initialValue;
        }

        @Override
        public Pooled<T> get() {
            Pooled<T> t = queue.poll();
            if (t != null)
                return t;

            if (created.incrementAndGet() > limit) {
                created.decrementAndGet();
                throw new IllegalStateException("Number of outstanding pooled objects has gone beyond the limit of " + limit);
            }
            return new Pooled<>(this, initialValue.get());
        }

        @Override
        public Stream<T> stream() {
            if (queue.size() < created.get())
                throw new IllegalStateException("close() was not called on all pooled objects yet");

            return queue.stream().filter(Objects::nonNull).map(Pooled::get);
        }

        @Override
        protected void onClosed(Pooled<T> value) {
            queue.offer(value);
        }
    }
}
