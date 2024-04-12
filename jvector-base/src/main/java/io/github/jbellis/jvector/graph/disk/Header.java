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

import java.io.DataOutput;
import java.io.IOException;
import java.util.EnumMap;
import java.util.EnumSet;

/**
 * Header information for an on-disk graph index, reflecting the common header and feature-specific headers.
 */
class Header {
    final CommonHeader common;
    final EnumMap<FeatureId, ? extends Feature> features;

    Header(CommonHeader common, EnumMap<FeatureId, ? extends Feature> features) {
        this.common = common;
        this.features = features;
    }

    void write(DataOutput out) throws IOException {
        out.writeInt(OnDiskGraphIndex.MAGIC);
        common.write(out);
        out.writeInt(FeatureId.serialize(EnumSet.copyOf(features.keySet())));
        for (Feature writer : features.values()) {
            writer.writeHeader(out);
        }
    }

    static Header load(RandomAccessReader reader, long offset) throws IOException {
        reader.seek(offset);
        int maybeMagic = reader.readInt();
        int version;
        EnumSet<FeatureId> featureIds;
        EnumMap<FeatureId, Feature> features = new EnumMap<>(FeatureId.class);
        CommonHeader common;

        if (maybeMagic != OnDiskGraphIndex.MAGIC) {
            // old format ODGI (DiskANN-style graph with inline full vectors)
            // maybeMagic contains size
            version = 0;
            featureIds = EnumSet.of(FeatureId.INLINE_VECTORS);
            int dimension = reader.readInt();
            int entryNode = reader.readInt();
            int maxDegree = reader.readInt();
            common = new CommonHeader(version, maybeMagic, dimension, entryNode, maxDegree);
        } else {
            common = CommonHeader.load(reader);
            featureIds = FeatureId.deserialize(reader.readInt());
        }

        for (FeatureId featureId : featureIds) {
            features.put(featureId, featureId.load(common, reader));
        }

        return new Header(common, features);
    }
}
