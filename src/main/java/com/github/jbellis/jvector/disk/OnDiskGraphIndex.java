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

import java.io.IOException;
import java.util.Arrays;

public class OnDiskGraphIndex<T> implements GraphIndex<T>, AutoCloseable
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

    @Override
    public int size() {
        return size;
    }

    public T getVector(int node) {
        // TODO
        throw new UnsupportedOperationException();
    }

    public int getNeighborCount(int node) {
        // TODO
        throw new UnsupportedOperationException();
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

        public Object getVector(int node) {
            // TODO
            return null;
        }

        public NodesIterator getNeighborsIterator(int node) {
            // TODO
            return null;
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

    public void close() throws Exception {
        readerSupplier.close();
    }
}
