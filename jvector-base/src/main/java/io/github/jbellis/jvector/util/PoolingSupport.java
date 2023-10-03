package io.github.jbellis.jvector.util;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.jctools.queues.MpmcArrayQueue;

public abstract class PoolingSupport<T> {

    public static <T> PoolingSupport<T> newThreadBased(Supplier<T> initialValue) {
        return new ThreadPooling<>(initialValue);
    }

    public static <T> PoolingSupport<T> newThreadBased(int threadLimit, Supplier<T> initialValue) {
        return new ThreadPooling<>(threadLimit, initialValue);
    }

    public static <T> PoolingSupport<T> newNoPooling(T fixedValue) {
        return new NoPooling<>(fixedValue);
    }


    private PoolingSupport() {
    }

    public abstract Pooled<T> get();

    public abstract Stream<T> stream();

    protected abstract void onClosed(T value);


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
        private final AtomicInteger created;
        private final MpmcArrayQueue<T> queue;
        private final Supplier<T> initialValue;

        private ThreadPooling(Supplier<T> initialValue) {
            //+1 for main thread
            this(Runtime.getRuntime().availableProcessors() + 1, initialValue);
        }

        private ThreadPooling(int threadLimit, Supplier<T> initialValue) {
            this.created = new AtomicInteger(0);
            this.queue = new MpmcArrayQueue<>(threadLimit);
            this.initialValue = initialValue;
        }

        public Pooled<T> get() {
            T t = queue.poll();
            if (t != null)
                return new Pooled<>(this, t);

            created.incrementAndGet();
            return new Pooled<>(this, initialValue.get());
        }

        public Stream<T> stream() {
            if (queue.size() < created.get())
                throw new RuntimeException("close() was not called on all pooled objects yet");

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
