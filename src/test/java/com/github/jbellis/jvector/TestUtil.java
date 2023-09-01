package com.github.jbellis.jvector;

import java.util.Random;

public class TestUtil {
    public static int nextInt(Random random, int min, int max) {
        return min + random.nextInt(max - min);
    }
}
