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

package io.github.jbellis.jvector.linalg;

import io.github.jbellis.jvector.vector.Matrix;

/**
 * Calculates the compact Singular Value Decomposition of a matrix using power iterations.
 * The Singular Value Decomposition of a m × n matrix A is a set of three matrices: U, S and V such that
 * A = U × S × V^T
 * where
 * - U is a m × k column-orthogonal matrix
 * - S is a k × k diagonal matrix with non-negative
 * - V is a k × n row-orthogonal matrix
 *
 * If k is not specified, k = min(m,n).
 */
public class SingularValueDecomposition {
    private final Matrix matrix;
    private final int k;

    SingularValueDecomposition(Matrix matrix, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be greater than zero");
        }
        this.matrix = matrix;
        this.k = k;
    }

    SingularValueDecomposition(Matrix matrix) {
        this.matrix = matrix;
        this.k = Math.min(matrix.getRowDimension(), matrix.getColumnDimension());
    }

    /** Returns the diagonal matrix S of the decomposition. */
    Matrix getS() {
        return null;
    }

    /** Returns the diagonal elements of the matrix S of the decomposition. */
    float[] getSingularValues() {
        return null;
    }

    /** Returns the matrix U of the decomposition. */
    Matrix	getU() {
        return null;
    }

    /** Returns the transpose of the matrix U of the decomposition. */
    Matrix	getUT() {
        return null;
    }

    /** Returns the matrix V of the decomposition. */
    Matrix	getV() {
        return null;
    }

    /** Returns the transpose of the matrix V of the decomposition. */
    Matrix	getVT() {
        return null;
    }
}
