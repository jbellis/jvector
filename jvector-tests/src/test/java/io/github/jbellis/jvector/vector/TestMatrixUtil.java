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

package io.github.jbellis.jvector.vector;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TestMatrixUtil {
    @Test
    public void testInvert() {
        var matrix = Matrix.from(new float[][] {{4, 7}, {2, 6}});
        var expected = Matrix.from(new float[][] {{0.6f, -0.7f}, {-0.2f, 0.4f}});
        assertEquals(expected, matrix.invert());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInvertNonSquareMatrix() {
        var matrix = Matrix.from(new float[][] {{1, 2, 3}, {4, 5, 6}});
        matrix.invert();
    }
}
