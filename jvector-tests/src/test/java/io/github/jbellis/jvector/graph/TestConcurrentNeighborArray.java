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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.FixedBitSet;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestConcurrentNeighborArray {
    @Test
    public void testRetainNoneSelected() {
        var array = new ConcurrentNeighborSet.ConcurrentNeighborArray(10, true);
        for (int i = 1; i <= 10; i++) {
            array.addInOrder(i, 11 - i);
        }
        var selected = new FixedBitSet(10); // All bits are false by default
        array.retain(selected);
        assertEquals(0, array.size());
    }

    @Test
    public void testRetainAllSelected() {
        var array = new ConcurrentNeighborSet.ConcurrentNeighborArray(10, true);
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
        var array = new ConcurrentNeighborSet.ConcurrentNeighborArray(10, true);
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
        var array = new ConcurrentNeighborSet.ConcurrentNeighborArray(10, true);
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
