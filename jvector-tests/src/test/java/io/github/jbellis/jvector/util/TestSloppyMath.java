package io.github.jbellis.jvector.util;

import org.junit.Test;

import static org.junit.Assert.assertThrows;
public class TestSloppyMath {
    @Test
    public void testLatLonBoundaries() {
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(-91, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(91, 0, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, -181, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 181, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0,-91, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 91, 0));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 0, -181));
        assertThrows(IllegalArgumentException.class, () -> SloppyMath.haversinMeters(0, 0, 0, 181));
    }
}
