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

#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "jvector_simd.h"

__m512i initialIndexRegister;
__m512i indexIncrement;

__attribute__((constructor))
void initialize_constants() {
    if (check_compatibility()) {
        initialIndexRegister = _mm512_setr_epi32(-16, -15, -14, -13, -12, -11, -10, -9,
                                             -8, -7, -6, -5, -4, -3, -2, -1);
        indexIncrement = _mm512_set1_epi32(16);
    }
}

float dot_product_f32_64(const float* a, int aoffset, const float* b, int boffset) {

     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_mul_ps(va, vb); // Perform element-wise multiplication

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float dot_product_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        dot = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
}

float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return dot_product_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return dot_product_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}

float euclidean_f32_64(const float* a, int aoffset, const float* b, int boffset) {
     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_sub_ps(va, vb);
     r = _mm_mul_ps(r, r);

    // Horizontal sum of the vector to get square distance
    __attribute__((aligned(8))) float result[2];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float euclidean_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 4);

    if (length >= 4) {
        __m128 sum = _mm_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 4, bo += 4) {
            // Load float32
            __m128 va = _mm_loadu_ps(a + ao);
            __m128 vb = _mm_loadu_ps(b + bo);
            __m128 diff = _mm_sub_ps(va, vb);
            // Multiply and accumulate
            sum = _mm_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[4];
        _mm_store_ps(result, sum);

        for(int i = 0; i < 4; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 8);

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < aoffset + simd_length; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);
            __m256 diff = _mm256_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        __attribute__((aligned(32))) float result[8];
        _mm256_store_ps(result, sum);

        for(int i = 0; i < 8; ++i) {
            squareDistance += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
    float squareDistance = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    int simd_length = length - (length % 16);

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();
        for(; ao < aoffset + simd_length; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);
            __m512 diff = _mm512_sub_ps(va, vb);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum of the vector to get dot product
        squareDistance = _mm512_reduce_add_ps(sum);
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        float diff = a[ao] - b[bo];
        squareDistance += diff * diff;
    }

    return squareDistance;
}

float euclidean_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return euclidean_f32_64(a, aoffset, b, boffset);
    if (length <= 7)
        return euclidean_f32_128(a, aoffset, b, boffset, length);

    return (preferred_size == 512 && length >= 16)
           ? euclidean_f32_512(a, aoffset, b, boffset, length)
           : euclidean_f32_256(a, aoffset, b, boffset, length);
}

void bulk_shuffle_dot_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results) {
    __m128i shuffleLeftRaw;
    __m128i shuffleRightRaw;
    __m512i shuffleLeft;
    __m512i shuffleRight;
    __m512 tmpLeft = _mm512_setzero_ps();
    __m512 tmpRight = _mm512_setzero_ps();

    for (int i = 0; i < codebookCount; i++) {
        shuffleLeftRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32));
        shuffleRightRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32 + 16));
        shuffleLeft = _mm512_cvtepu8_epi32(shuffleLeftRaw);
        shuffleRight = _mm512_cvtepu8_epi32(shuffleRightRaw);
        __m512 partialsVecA = _mm512_loadu_ps(partials + i * 32);
        __m512 partialsVecB = _mm512_loadu_ps(partials + i * 32 + 16);
        // use permutex2var_ps
        __m512 partialsVec = _mm512_permutex2var_ps(partialsVecA, shuffleLeft, partialsVecB);
        tmpLeft = _mm512_add_ps(tmpLeft, partialsVec);
        partialsVec = _mm512_permutex2var_ps(partialsVecA, shuffleRight, partialsVecB);
        tmpRight = _mm512_add_ps(tmpRight, partialsVec);
    }

    tmpLeft = _mm512_add_ps(tmpLeft, _mm512_set1_ps(1.0));
    tmpRight = _mm512_add_ps(tmpRight, _mm512_set1_ps(1.0));
    tmpLeft = _mm512_div_ps(tmpLeft, _mm512_set1_ps(2.0));
    tmpRight = _mm512_div_ps(tmpRight, _mm512_set1_ps(2.0));

    _mm512_storeu_ps(results, tmpLeft);
    _mm512_storeu_ps(results + 16, tmpRight);
}

void bulk_shuffle_euclidean_f32_512(const unsigned char* shuffles, int codebookCount, const float* partials, float* results) {
    __m128i shuffleLeftRaw;
    __m128i shuffleRightRaw;
    __m512i shuffleLeft;
    __m512i shuffleRight;
    __m512 tmpLeft = _mm512_setzero_ps();
    __m512 tmpRight = _mm512_setzero_ps();

    for (int i = 0; i < codebookCount; i++) {
        shuffleLeftRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32));
        shuffleRightRaw = _mm_loadu_si128((__m128i *)(shuffles + i * 32 + 16));
        shuffleLeft = _mm512_cvtepu8_epi32(shuffleLeftRaw);
        shuffleRight = _mm512_cvtepu8_epi32(shuffleRightRaw);
        __m512 partialsVecA = _mm512_loadu_ps(partials + i * 32);
        __m512 partialsVecB = _mm512_loadu_ps(partials + i * 32 + 16);
        __m512 partialsVec = _mm512_permutex2var_ps(partialsVecA, shuffleLeft, partialsVecB);
        tmpLeft = _mm512_add_ps(tmpLeft, partialsVec);
        partialsVec = _mm512_permutex2var_ps(partialsVecA, shuffleRight, partialsVecB);
        tmpRight = _mm512_add_ps(tmpRight, partialsVec);
    }

    // add 1 to tmpLeft/tmpRight
    __m512 ones = _mm512_set1_ps(1.0);
    tmpLeft = _mm512_add_ps(tmpLeft, ones);
    tmpRight = _mm512_add_ps(tmpRight, ones);
    // reciprocal of tmpLeft/tmpRight
    tmpLeft = _mm512_rcp14_ps(tmpLeft);
    tmpRight = _mm512_rcp14_ps(tmpRight);
    _mm512_storeu_ps(results, tmpLeft);
    _mm512_storeu_ps(results + 16, tmpRight);
}

float assemble_and_sum_f32_512(const float* data, int dataBase, const unsigned char* baseOffsets, int baseOffsetsLength) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    int limit = baseOffsetsLength - (baseOffsetsLength % 16);
    __m512i indexRegister = initialIndexRegister;
    __m512i dataBaseVec = _mm512_set1_epi32(dataBase);

    for (; i < limit; i += 16) {
        __m128i baseOffsetsRaw = _mm_loadu_si128((__m128i *)(baseOffsets + i));
        __m512i baseOffsetsInt = _mm512_cvtepu8_epi32(baseOffsetsRaw);
        // we have base offsets int, which we need to scale to index into data.
        // first, we want to initialize a vector with the lane number added as an index
        indexRegister = _mm512_add_epi32(indexRegister, indexIncrement);
        // then we want to multiply by dataBase
        __m512i scale = _mm512_mullo_epi32(indexRegister, dataBaseVec);
        // then we want to add the base offsets
        __m512i convOffsets = _mm512_add_epi32(scale, baseOffsetsInt);

        __m512 partials = _mm512_i32gather_ps(convOffsets, data, 4);
        sum = _mm512_add_ps(sum, partials);
    }

    float res = _mm512_reduce_add_ps(sum);
    for (; i < baseOffsetsLength; i++) {
        res += data[dataBase * i + baseOffsets[i]];
    }

    return res;
}

void calculate_partial_sums_dot_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = dot_product_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

void calculate_partial_sums_euclidean_f32_512(const float* codebook, int codebookBase, int size, int clusterCount, const float* query, int queryOffset, float* partialSums) {
    for (int i = 0; i < clusterCount; i++) {
      partialSums[codebookBase + i] = euclidean_f32(512, codebook, i * size, query, queryOffset, size);
    }
}

void dot_product_multi_f32_512(const float* v1, const float* packedv2, int v1Length, int resultsLength, float* results) {
    int ao = 0;
    int simd_length = v1Length - (v1Length % 16);


    if (v1Length >= 16) {
        __m512 sums[resultsLength]; // Array of sums for each subvector in c
        for (int k = 0; k < resultsLength; ++k) {
            sums[k] = _mm512_setzero_ps();
        }

        for (; ao < simd_length; ao += 16) {
            __m512 va = _mm512_loadu_ps(v1 + ao);

            for (int k = 0; k < resultsLength; ++k) {
                // Load float32 from the k-th subvector of c
                __m512 vc = _mm512_loadu_ps(packedv2 + ao + (k * v1Length));
                // Multiply and accumulate for the k-th subvector
                sums[k] = _mm512_fmadd_ps(va, vc, sums[k]);
            }
        }

        // Horizontal sum of the vectors to get K dot products
        for (int k = 0; k < resultsLength; ++k) {
            results[k] = _mm512_reduce_add_ps(sums[k]);
        }
    }

    // Scalar computation for remaining elements
    for (; ao < v1Length; ao++) {
        for (int k = 0; k < resultsLength; ++k) {
            results[k] += v1[ao] * packedv2[ao + (k * v1Length)];
        }
    }

    // convert to scores
    for (int k = 0; k < resultsLength; ++k) {
        results[k] = (1.0f + results[k] ) / 2;
    }
}

void square_distance_multi_f32_512(const float* v1, const float* packedv2, int v1Length, int resultsLength, float* results) {
    int ao = 0;
    int simd_length = v1Length - (v1Length % 16);


    if (v1Length >= 16) {
        __m512 sums[resultsLength]; // Array of sums for each subvector in c
        for (int k = 0; k < resultsLength; ++k) {
            sums[k] = _mm512_setzero_ps();
        }

        for (; ao < simd_length; ao += 16) {
            __m512 va = _mm512_loadu_ps(v1 + ao);

            for (int k = 0; k < resultsLength; ++k) {
                // Load float32 from the k-th subvector of c
                __m512 vc = _mm512_loadu_ps(packedv2 + ao + (k * v1Length));
                // Multiply and accumulate for the k-th subvector
                __m512 diff = _mm512_sub_ps(va, vc);
                sums[k] = _mm512_fmadd_ps(diff, diff, sums[k]);
            }
        }

        // Horizontal sum of the vectors to get K dot products
        for (int k = 0; k < resultsLength; ++k) {
            results[k] = _mm512_reduce_add_ps(sums[k]);
        }
    }

    // Scalar computation for remaining elements
    for (; ao < v1Length; ao++) {
        for (int k = 0; k < resultsLength; ++k) {
            float diff = v1[ao] - packedv2[ao + (k * v1Length)];
            results[k] += diff * diff;
        }
    }

    // convert to scores
    for (int k = 0; k < resultsLength; ++k) {
        results[k] = 1.0f / (1 + results[k]);
    }
}
