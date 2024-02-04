package io.github.jbellis.jvector.util;

import org.agrona.collections.Long2ObjectHashMap;

import java.util.function.Function;
import java.util.function.Supplier;

/**
 * A replacement for {@link ThreadLocal} that doesn't play clever tricks with the ThreadLocalMap,
 * which means that we can guarantee that the values are GC'd once the parent instance is no
 * longer referenced.
 */
public abstract class ExplicitThreadLocal<U> {
    private final Long2ObjectHashMap<U> map = new Long2ObjectHashMap<>();

    // computeIfAbsent wants a callable that takes a parameter, but if we use a lambda
    // it will be a closure and we'll get a new instance for every call.  So we instantiate
    // it just once here as a field instead.
    private final Function<Long, U> initialSupplier = k -> initialValue();

    public U get() {
        return map.computeIfAbsent(Thread.currentThread().getId(), initialSupplier);
    }

    protected abstract U initialValue();

    public static <U> ExplicitThreadLocal<U> withInitial(Supplier<U> initialValue) {
        return new ExplicitThreadLocal<>() {
            @Override
            protected U initialValue() {
                return initialValue.get();
            }
        };
    }
}

