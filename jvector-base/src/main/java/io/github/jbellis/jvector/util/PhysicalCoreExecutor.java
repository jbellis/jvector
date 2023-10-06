package io.github.jbellis.jvector.util;

import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.pq.ProductQuantization;

/**
 * A fork join pool which is sized to match the number of physical cores on the machine (avoiding hyper-thread count)
 *
 * This is important for heavily vectorized sections of the code since it can easily saturate memory bandwidth.
 *
 * @see ProductQuantization
 * @see GraphIndexBuilder
 *
 * Knowing how many physical cores a machine has is left to the operator (however the default of 1/2 cores is today often correct).
 */
public class PhysicalCoreExecutor {
    private static final int physicalCoreCount = Integer.getInteger("jvector.physical_core_count", Math.max(1, Runtime.getRuntime().availableProcessors()/2));

    public static final PhysicalCoreExecutor instance = new PhysicalCoreExecutor(physicalCoreCount);

    private final ForkJoinPool pool;

    private PhysicalCoreExecutor(int cores) {
        assert cores > 0 && cores <= Runtime.getRuntime().availableProcessors() : "Invalid core count: " + cores;
        this.pool = new ForkJoinPool(cores);
    }

    public void execute(Runnable run) {
        pool.submit(run).join();
    }

    public <T> T submit(Supplier<T> run) {
        return pool.submit(run::get).join();
    }
}
