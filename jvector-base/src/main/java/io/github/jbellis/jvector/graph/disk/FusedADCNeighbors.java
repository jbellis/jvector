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

import io.github.jbellis.jvector.vector.types.ByteSequence;

public interface FusedADCNeighbors {
    /**
     * Get the ADC-packed neighbor vectors of the given node.
     * @param node the node to get the neighbors of.
     * @return the quantized, transposed, and packed vectors of the given node's neighbors.
     */
    ByteSequence<?> getPackedNeighbors(int node);

    /**
     * Get the maximum degree of the graph for which this FusedADCNeighbors instance is used.
     * @return the maximum degree of the graph.
     */
    int maxDegree();
}
