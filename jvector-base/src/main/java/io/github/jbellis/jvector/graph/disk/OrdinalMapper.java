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

package io.github.jbellis.jvector.graph.disk;

import org.agrona.collections.Int2IntHashMap;

import java.util.Map;

public interface OrdinalMapper {
    int OMITTED = Integer.MIN_VALUE;

    int maxOrdinal();

    int oldToNew(int oldOrdinal);

    int newToOld(int newOrdinal);

    class IdentityMapper implements OrdinalMapper {
        private final int maxOrdinal;

        public IdentityMapper(int maxOrdinal) {
            this.maxOrdinal = maxOrdinal;
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldOrdinal;
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newOrdinal;
        }
    }

    class MapMapper implements OrdinalMapper {
        private final int maxOrdinal;
        private final Map<Integer, Integer> oldToNew;
        private final Int2IntHashMap newToOld;

        public MapMapper(Map<Integer, Integer> oldToNew) {
            this.oldToNew = oldToNew;
            this.newToOld = new Int2IntHashMap(oldToNew.size(), 0.65f, OMITTED);
            oldToNew.forEach((old, newOrdinal) -> newToOld.put(newOrdinal, old));
            this.maxOrdinal = oldToNew.values().stream().mapToInt(i -> i).max().orElse(-1);
        }

        @Override
        public int maxOrdinal() {
            return maxOrdinal;
        }

        @Override
        public int oldToNew(int oldOrdinal) {
            return oldToNew.get(oldOrdinal);
        }

        @Override
        public int newToOld(int newOrdinal) {
            return newToOld.get(newOrdinal);
        }
    }
}
