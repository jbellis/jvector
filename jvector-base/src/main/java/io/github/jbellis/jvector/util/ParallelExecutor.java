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

import java.util.concurrent.ForkJoinTask;
import java.util.function.Supplier;

/**
 * An executor to run tasks in parallel inside a {@code ForkJoinPool}.
 * <p>
 * If the caller is already in a {@code ForkJoinPool},
 * then the given task is run on the caller thread.
 * Otherwise, task is submitted to {@code PhysicalCoreExecutor}.
 *
 * @see PhysicalCoreExecutor
 */
public class ParallelExecutor {

    public static void execute(Runnable run) {
        if (ForkJoinTask.inForkJoinPool()) {
            run.run();
        } else {
            PhysicalCoreExecutor.instance.execute(run);
        }
    }

    public static <T> T submit(Supplier<T> run) {
        if (ForkJoinTask.inForkJoinPool()) {
            return run.get();
        } else {
            return PhysicalCoreExecutor.instance.submit(run);
        }
    }
}
