/*
 * All changes to the original code are Copyright DataStax, Inc.
 *
 * Please see the included license file for details.
 */

/*
 * Original license:
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.util;

import io.github.jbellis.jvector.LuceneTestCase;
import io.github.jbellis.jvector.TestUtil;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.Assert.*;

public class TestLongHeap extends LuceneTestCase {

    private static void checkValidity(AbstractLongHeap heap) {
        long[] heapArray = heap.getHeapArray();
        for (int i = 2; i <= heap.size(); i++) {
            int parent = i >>> 1;
            assert heapArray[parent] <= heapArray[i];
        }
    }

    @Test
    public void testPQ() {
        testPQ(atLeast(1000), random());
    }

    public static void testPQ(int count, Random gen) {
        var pq = new GrowableLongHeap(count);
        long sum = 0, sum2 = 0;

        for (int i = 0; i < count; i++) {
            long next = gen.nextLong();
            sum += next;
            pq.push(next);
        }

        long last = Long.MIN_VALUE;
        for (long i = 0; i < count; i++) {
            long next = pq.pop();
            assertTrue(next >= last);
            last = next;
            sum2 += last;
        }

        assertEquals(sum, sum2);
    }

    @Test
    public void testClear() {
        var pq = new GrowableLongHeap(3);
        pq.push(2);
        pq.push(3);
        pq.push(1);
        assertEquals(3, pq.size());
        pq.clear();
        assertEquals(0, pq.size());
    }

    @Test
    public void testExceedBounds() {
        var pq = new GrowableLongHeap(1);
        pq.push(2);
        pq.push(0);
        assertEquals(2, pq.size()); // the heap has been extended to a new max size
        assertEquals(0, pq.top());
    }

    @Test
    public void testFixedSize() {
        var pq = new BoundedLongHeap(3);
        pq.push(2);
        pq.push(3);
        pq.push(1);
        pq.push(5);
        pq.push(7);
        pq.push(1);
        assertEquals(3, pq.size());
        assertEquals(3, pq.top());
    }

    @Test
    public void testDuplicateValues() {
        var pq = new BoundedLongHeap(3);
        pq.push(2);
        pq.push(3);
        pq.push(1);
        assertEquals(1, pq.top());
        pq.updateTop(3);
        assertEquals(3, pq.size());
        assertArrayEquals(new long[]{0, 2, 3, 3}, pq.getHeapArray());
    }

    @Test
    public void testInsertions() {
        Random random = random();
        int numDocsInPQ = TestUtil.nextInt(random, 1, 100);
        var pq = new BoundedLongHeap(numDocsInPQ);
        Long lastLeast = null;

        // Basic insertion of new content
        var sds = new ArrayList<Long>(numDocsInPQ);
        for (int i = 0; i < numDocsInPQ * 10; i++) {
            long newEntry = Math.abs(random.nextLong());
            sds.add(newEntry);
            pq.push(newEntry);
            checkValidity(pq);
            long newLeast = pq.top();
            if ((lastLeast != null) && (newLeast != newEntry) && (newLeast != lastLeast)) {
                // If there has been a change of least entry and it wasn't our new
                // addition we expect the scores to increase
                assertTrue(newLeast <= newEntry);
                assertTrue(newLeast >= lastLeast);
            }
            lastLeast = newLeast;
        }
    }

    @Test
    public void testInvalid() {
        assertThrows(IllegalArgumentException.class, () -> new GrowableLongHeap(-1));
        assertThrows(IllegalArgumentException.class, () -> new GrowableLongHeap(0));
        assertThrows(IllegalArgumentException.class, () -> new GrowableLongHeap(ArrayUtil.MAX_ARRAY_LENGTH));
    }
}
