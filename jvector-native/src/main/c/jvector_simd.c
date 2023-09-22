#include <stdio.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "jvector_simd.h"

float dot_product_f32_64(const float* a, int aoffset, const float* b, int boffset) {

     __m128 va = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(a + aoffset)));
     __m128 vb = _mm_castsi128_ps(_mm_loadl_epi64((__m128i *)(b + boffset)));
     __m128 r  = _mm_mul_ps(va, vb); // Perform element-wise multiplication

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, r);
    return result[0] + result[1];
}

float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    if (length >= 8) {
        __m256 sum = _mm256_setzero_ps();

        for(; ao < alim && bo < blim; ao += 8, bo += 8) {
            // Load float32
            __m256 va = _mm256_loadu_ps(a + ao);
            __m256 vb = _mm256_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16))) float result[8];
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
#if defined(__AVX512F__)
    float dot = 0.0;
    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    if (length >= 16) {
        __m512 sum = _mm512_setzero_ps();

        for(; ao < alim && bo < blim; ao += 16, bo += 16) {
            // Load float32
            __m512 va = _mm512_loadu_ps(a + ao);
            __m512 vb = _mm512_loadu_ps(b + bo);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum of the vector to get dot product
        __attribute__((aligned(16)))  float result[16];
        _mm512_store_ps(result, sum);

        for(int i = 0; i < 16; ++i) {
            dot += result[i];
        }
    }

    for (; ao < alim && bo < blim; ao++, bo++) {
        dot += a[ao] * b[bo];
    }

    return dot;
#else
    return dot_product_f32_256(a, aoffset, b, boffset, length);
#endif
}

float dot_product_f32(int preferred_size, const float* a, int aoffset, const float* b, int boffset, int length) {
    if (length == 2)
        return dot_product_f32_64(a, aoffset, b, boffset);

    return (preferred_size == 512 && length >= 16)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}


