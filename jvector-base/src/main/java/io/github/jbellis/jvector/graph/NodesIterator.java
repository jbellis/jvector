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

package io.github.jbellis.jvector.graph;

import java.util.NoSuchElementException;
import java.util.PrimitiveIterator;

/**
 * Iterator over graph nodes that includes the size –- the total
 * number of nodes to be iterated over. The nodes are NOT guaranteed to be presented in any
 * particular order.
 */
public abstract class NodesIterator implements PrimitiveIterator.OfInt {
    protected final int size;

    /**
     * Constructor for iterator based on the size
     */
    public NodesIterator(int size) {
        this.size = size;
    }

    /**
     * The number of elements in this iterator *
     */
    public int size() {
        return size;
    }

    public static NodesIterator fromPrimitiveIterator(PrimitiveIterator.OfInt iterator, int size) {
        return new NodesIterator(size) {
            @Override
            public int nextInt() {
                return iterator.nextInt();
            }

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }
        };
    }

    public static class ArrayNodesIterator extends NodesIterator {
        private final int[] nodes;
        private int cur;

        /** Constructor for iterator based on integer array representing nodes */
        public ArrayNodesIterator(int[] nodes, int offset, int count) {
            super(count);
            assert nodes != null;
            assert offset + size <= nodes.length;
            this.nodes = nodes;
            this.cur = offset;
        }

        public ArrayNodesIterator(int[] nodes, int count) {
            this(nodes, 0, count);
        }

        public ArrayNodesIterator(int[] nodes) {
            this(nodes, nodes.length);
        }

        @Override
        public int nextInt() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return nodes[cur++];
        }

        @Override
        public boolean hasNext() {
            return cur < nodes.length;
        }
    }
}
