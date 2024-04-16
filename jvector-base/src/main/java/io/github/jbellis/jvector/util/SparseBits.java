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

package io.github.jbellis.jvector.util;

import org.agrona.collections.IntHashSet;

/**
 * Implements the membership parts of an updatable BitSet (but not prev/next bits)
 */
public class SparseBits implements Bits {
    private final IntHashSet set = new IntHashSet();

    @Override
    public boolean get(int index) {
        return set.contains(index);
    }

    public void set(int index) {
        set.add(index);
    }

    public void clear() {
        set.clear();
    }

    public int cardinality() {
        return set.size();
    }
}
