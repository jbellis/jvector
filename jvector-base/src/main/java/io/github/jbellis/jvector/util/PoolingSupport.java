package io.github.jbellis.jvector.util;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.jctools.queues.MpmcArrayQueue;

/**
 * Allows any object to be pooled and released when work is done.
 * This is an alternative to using {@link ThreadLocal}.
 *
 * @param <T> The object to be pooled by this instance
 */
public abstract class PoolingSupport<T> {

    /**
     * Creates a pool of objects intended to be used by a fixed thread pool.
     * The pool size will the processor count.
     * @param initialValue allows creation of new instances for the pool
     */
    public static <T> PoolingSupport<T> newThreadBased(Supplier<T> initialValue) {
        return new ThreadPooling<>(initialValue);
    }

    /**
     * Creates a pool intended to be used by a fixed thread pool
     * @param threadLimit the specific number of threads to be sharing the pooled objects
     * @param initialValue allows creation of new instances for the pool
     */
    public static <T> PoolingSupport<T> newThreadBased(int threadLimit, Supplier<T> initialValue) {
        return new ThreadPooling<>(threadLimit, initialValue);
    }

    /**
     * Special case of not actually needing a pool (when other times you do)
     *
     * @param fixedValue the value this pool will always return
     */
    public static <T> PoolingSupport<T> newNoPooling(T fixedValue) {
        return new NoPooling<>(fixedValue);
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
    protected abstract void onClosed(T value);

    /**
     * Wrapper class for items in the pool
     *
     * These are AutoClosable and are intended to be used
     * in a try-with-resources statement.
     * @param <T>
     */
    public static class Pooled<T> implements AutoCloseable {
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
            owner.onClosed(this.value);
        }
    }


    static class ThreadPooling<T> extends PoolingSupport<T>
    {
        private final int limit;
        private final AtomicInteger created;
        private final MpmcArrayQueue<T> queue;
        private final Supplier<T> initialValue;

        private ThreadPooling(Supplier<T> initialValue) {
            //+1 for main thread
            this(Runtime.getRuntime().availableProcessors() + 1, initialValue);
        }

        private ThreadPooling(int threadLimit, Supplier<T> initialValue) {
            this.limit = threadLimit;
            this.created = new AtomicInteger(0);
            this.queue = new MpmcArrayQueue<>(threadLimit);
            this.initialValue = initialValue;
        }

        public Pooled<T> get() {
            T t = queue.poll();
            if (t != null)
                return new Pooled<>(this, t);

            if (created.incrementAndGet() > limit) {
                created.decrementAndGet();
                throw new IllegalStateException("Number of outstanding pooled objects has gone beyond the limit of " + limit);
            }
            return new Pooled<>(this, initialValue.get());
        }

        public Stream<T> stream() {
            if (queue.size() < created.get())
                throw new IllegalStateException("close() was not called on all pooled objects yet");

            return queue.stream();
        }

        protected void onClosed(T value) {
            queue.offer(value);
        }
    }


    static class NoPooling<T> extends PoolingSupport<T> {
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
        protected void onClosed(T value) {
        }
    }
}
