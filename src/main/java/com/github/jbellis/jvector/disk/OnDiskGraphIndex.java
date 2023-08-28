/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.jbellis.jvector.disk;

import com.github.jbellis.jvector.annotations.VisibleForTesting;
import com.github.jbellis.jvector.graph.GraphIndex;
import com.github.jbellis.jvector.graph.NodesIterator;

import static com.github.jbellis.jvector.graph.NodesIterator.NO_MORE_NEIGHBORS;

import java.util.Arrays;

public class OnDiskGraphIndex implements GraphIndex, AutoCloseable
{
    private final ReaderSupplier readerSupplier;
    private final long segmentOffset;
    private final int size;
    private final int entryNode;
    private final int M;
    private final int dimension;

    public OnDiskGraphIndex(ReaderSupplier readerSupplier, long offset)
    {
        this.readerSupplier = readerSupplier;
        this.segmentOffset = offset;
        try (var reader = readerSupplier.get()) {
            reader.seek(offset);
            size = reader.readInt();
            entryNode = reader.readInt();
            M = reader.readInt();
            dimension = reader.readInt();
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    private static long levelNodeOf(int level, int target)
    {
        return ((long) level << 32) | target;
    }

    @Override
    public int size() {
        return size;
    }

    /** return an Graph that can be safely querried concurrently */
    public OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    public class OnDiskView implements GraphIndex.View, AutoCloseable
    {
        private final RandomAccessReader reader;
        private int currentNeighborsRead;
        private long currentCachedLevelNode = -1;
        private int[] currentNeighbors;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
        }

        @Override
        public void seek(int target) {
            // TODO
        }

        @Override
        public int nextNeighbor()
        {
            if (currentNeighborsRead++ < currentNeighbors.length)
            {
                return currentNeighbors[currentNeighborsRead - 1];
            }
            return NO_MORE_NEIGHBORS;
        }

        @Override
        public int size() {
            return OnDiskGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return OnDiskGraphIndex.this.entryNode;
        }

        @Override
        public void close()
        {
            reader.close();
        }
    }

    @Override
    public NodesIterator getNodes()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public void addNode(int node) {
        throw new UnsupportedOperationException();
    }

    public void close() throws Exception {
        readerSupplier.close();
    }

    @VisibleForTesting
    static class CachedLevel
    {
        private final int level;
        private final int[] nodeIds;
        private final long[] offsets;
        private final int[][] neighbors;

        public CachedLevel(int level, int[] nodeIds, long[] offsets)
        {
            this.level = level;
            this.nodeIds = nodeIds;
            this.offsets = offsets;
            this.neighbors = null;
        }

        public CachedLevel(int level, int[] nodeIds, int[][] neighbors)
        {
            this.level = level;
            this.nodeIds = nodeIds;
            this.neighbors = neighbors;
            offsets = null;
        }

        public boolean containsNeighbors() {
            return neighbors != null;
        }

        public long offsetFor(int nodeId)
        {
            int i = Arrays.binarySearch(nodeIds, nodeId);
            if (i < 0)
                throw new IllegalStateException("Node " + nodeId + " not found in level " + level);
            return offsets[i];
        }

        public int[] neighborsFor(int nodeId)
        {
            int i = Arrays.binarySearch(nodeIds, nodeId);
            if (i < 0)
                throw new IllegalStateException("Node " + nodeId + " not found in level " + level);
            return neighbors[i];
        }

        public int[] nodesOnLevel()
        {
            return nodeIds;
        }
    }
}


