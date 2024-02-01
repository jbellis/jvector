package io.github.jbellis.jvector.vector;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TestMatrixUtil {
    @Test
    public void testInvert() {
        var matrix = Matrix.from(new float[][] {{4, 7}, {2, 6}});
        var expected = Matrix.from(new float[][] {{0.6f, -0.7f}, {-0.2f, 0.4f}});
        assertEquals(expected, matrix.invert());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvertNonSquareMatrix() {
        var matrix = Matrix.from(new float[][] {{1, 2, 3}, {4, 5, 6}});
        matrix.invert();
    }
}
