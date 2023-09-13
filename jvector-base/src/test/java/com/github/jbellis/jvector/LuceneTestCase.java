package com.github.jbellis.jvector;

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
