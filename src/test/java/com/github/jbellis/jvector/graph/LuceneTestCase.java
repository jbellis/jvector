package com.github.jbellis.jvector.graph;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class LuceneTestCase {
    // TODO make this deterministic
    public static Random random() {
        return ThreadLocalRandom.current();
    }
}
