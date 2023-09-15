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

package io.github.jbellis.jvector;

import com.carrotsearch.randomizedtesting.RandomizedTest;

import java.util.Random;

// Not actually derived from Lucene, but provides a random() method like LuceneTestCase does
// for easier porting of Lucene tests
public class LuceneTestCase extends RandomizedTest {
    public static int RANDOM_MULTIPLIER = 2;

    public static Random random() {
        return getRandom();
    }

    public static int atLeast(Random random, int n) {
        return n + random.nextInt(n / 2);
    }

    public static int atLeast(int n) {
        return n + getRandom().nextInt(n / 2);
    }

    public static int nextInt(int from, int to) {
        return getRandom().nextInt(to - from) + from;
    }
}
