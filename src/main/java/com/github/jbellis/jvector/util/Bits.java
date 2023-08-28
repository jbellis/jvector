/*
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
package com.github.jbellis.jvector.util;

/**
 * Interface for Bitset-like structures.
 *
 * @lucene.experimental
 */
public interface Bits {
    /**
     * Returns the value of the bit with the specified <code>index</code>.
     *
     * @param index index, should be non-negative and &lt; {@link #length()}. The result of passing
     *     negative or out of bounds values is undefined by this interface, <b>just don't do it!</b>
     * @return <code>true</code> if the bit is set, <code>false</code> otherwise.
     */
    boolean get(int index);

    /** Returns the number of bits in this set */
    int length();
}
