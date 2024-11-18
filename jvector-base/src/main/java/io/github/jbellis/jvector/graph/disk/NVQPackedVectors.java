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

import io.github.jbellis.jvector.pq.NVQuantization;

public interface NVQPackedVectors {
    /**
     * Get the NVQ quantized vector for the given ordinal.
     * All returned vectors are shared and will only be valid until the next call to this method on this object.
     * @param ordinal the ordinal of the vector to get
     * @return the NVQ quantized vector
     */
    NVQuantization.QuantizedVector getQuantizedVector(int ordinal);
}