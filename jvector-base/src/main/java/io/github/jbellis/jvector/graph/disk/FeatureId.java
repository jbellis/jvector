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

import io.github.jbellis.jvector.disk.RandomAccessReader;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * An enum representing the features that can be stored in an on-disk graph index.
 * Order of this enum is critical! It will be serialized on-disk in enum order.
 * These are typically mapped to a Feature or FeatureWriter.
 */
public enum FeatureId {
    INLINE_VECTORS(InlineVectors::load),
    FUSED_ADC(FusedADC::load),
    LVQ(io.github.jbellis.jvector.graph.disk.LVQ::load);

    public static Set<FeatureId> ALL = Collections.unmodifiableSet(EnumSet.allOf(FeatureId.class));

    private final BiFunction<CommonHeader, RandomAccessReader, Feature> loader;

    FeatureId(BiFunction<CommonHeader, RandomAccessReader, Feature> loader) {
        this.loader = loader;
    }

    public Feature load(CommonHeader header, RandomAccessReader reader) {
        return loader.apply(header, reader);
    }

    public static EnumSet<FeatureId> deserialize(int features) {
        EnumSet<FeatureId> set = EnumSet.noneOf(FeatureId.class);
        for (int n = 0; n < values().length; n++) {
            if ((features & (1 << n)) != 0)
                set.add(values()[n]);
        }
        return set;
    }

    public static int serialize(EnumSet<FeatureId> flags) {
        int i = 0;
        for (FeatureId flag : flags)
            i |= 1 << flag.ordinal();
        return i;
    }
}
