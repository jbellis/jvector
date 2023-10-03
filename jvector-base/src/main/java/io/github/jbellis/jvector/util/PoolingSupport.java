package io.github.jbellis.jvector.util;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.jctools.queues.MpmcArrayQueue;

public class PoolingSupport<T> {

    public static class Pooled<T> implements AutoCloseable {
        private final T value;
        private final PoolingSupport<T> owner;
        private Pooled(PoolingSupport owner, T value) {
            this.owner = owner;
            this.value = value;
        }

        public T get() {
            return value;
        }

        @Override
        public void close() {
            owner.queue.offer(this.value);
        }
    }

    private final AtomicInteger created;
    private final MpmcArrayQueue<T> queue;
    private final Supplier<T> initialValue;

    public PoolingSupport(Supplier<T> initialValue) {
        //+1 for main thread
        this(Runtime.getRuntime().availableProcessors() + 1, initialValue);
    }

    public PoolingSupport(int threadLimit, Supplier<T> initialValue) {
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
}
