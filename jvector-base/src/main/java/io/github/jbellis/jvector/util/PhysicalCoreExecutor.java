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

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.quantization.ProductQuantization;

import java.io.Closeable;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;

/**
 * A fork join pool which is sized to match the number of physical cores on the machine (avoiding hyper-thread count)
 * <p>
 * This is important for heavily vectorized sections of the code since it can easily saturate memory bandwidth.
 *
 * @see ProductQuantization
 * @see GraphIndexBuilder
 *
 * Knowing how many physical cores a machine has is left to the operator (however the default of 1/2 cores is today often correct).
 */
public class PhysicalCoreExecutor implements Closeable {
    private static final int physicalCoreCount = Integer.getInteger("jvector.physical_core_count", Math.max(1, Runtime.getRuntime().availableProcessors()/2));

    public static final PhysicalCoreExecutor instance = new PhysicalCoreExecutor(physicalCoreCount);

    public static ForkJoinPool pool() {
        return instance.pool;
    }
    
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

    public static int getPhysicalCoreCount() {
        return physicalCoreCount;
    }

    @Override
    public void close() {
        pool.shutdownNow();
    }
}
