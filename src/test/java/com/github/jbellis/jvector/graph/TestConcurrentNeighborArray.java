package com.github.jbellis.jvector.graph;

import org.junit.Test;

import com.github.jbellis.jvector.graph.ConcurrentNeighborSet.ConcurrentNeighborArray;
import com.github.jbellis.jvector.util.BitSet;
import com.github.jbellis.jvector.util.FixedBitSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestConcurrentNeighborArray {
    @Test
    public void testRetainNoneSelected() {
        var array = new ConcurrentNeighborArray(10, true);
        for (int i = 1; i <= 10; i++) {
            array.addInOrder(i, 11 - i);
        }
        var selected = new FixedBitSet(10); // All bits are false by default
        array.retain(selected);
        assertEquals(0, array.size());
    }

    @Test
    public void testRetainAllSelected() {
        var array = new ConcurrentNeighborArray(10, true);
        for (int i = 1; i <= 10; i++) {
            array.addInOrder(i, 11 - i);
        }
        var selected = new FixedBitSet(10);
        selected.set(0, 10); // Set all bits to true
        array.retain(selected);
        assertEquals(10, array.size());
    }

    @Test
    public void testRetainSomeSelectedNotFront() {
        var array = new ConcurrentNeighborArray(10, true);
        for (int i = 1; i <= 10; i++) {
            array.addInOrder(i, 11 - i);
        }
        var selected = new FixedBitSet(10);
        selected.set(5, 10); // Select last 5 elements
        array.retain(selected);
        assertEquals(5, array.size());
        for (int i = 0; i < array.size(); i++) {
            assertTrue(selected.get(i + 5));
        }
    }

    @Test
    public void testRetainSomeSelectedAtFront() {
        var array = new ConcurrentNeighborArray(10, true);
        for (int i = 1; i <= 10; i++) {
            array.addInOrder(i, 11 - i);
        }
        var selected = new FixedBitSet(10);
        selected.set(0, 5); // Select first 5 elements
        array.retain(selected);
        assertEquals(5, array.size());
        for (int i = 0; i < array.size(); i++) {
            assertTrue(selected.get(i));
        }
    }
}
