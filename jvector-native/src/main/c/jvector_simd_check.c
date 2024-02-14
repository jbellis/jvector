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

#include <cpuid.h>
#include "jvector_simd.h"

bool check_compatibility() {
    unsigned int eax, ebx, ecx, edx;
    bool avx512f_supported = 0, avx512cd_supported = 0, avx512bw_supported = 0, avx512dq_supported = 0, avx512vl_supported = 0;

    // Check for AVX-512 Foundation (AVX-512F) support:
    // It is indicated by bit 16 of EBX from leaf 7, sub-leaf 0.
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        avx512f_supported = ebx & (1 << 16);
    }

    // Check for AVX-512 Conflict Detection (AVX-512CD) support:
    // It is indicated by bit 28 of EBX from leaf 7, sub-leaf 0.
    avx512cd_supported = ebx & (1 << 28);

    // Check for AVX-512 Byte and Word Instructions (AVX-512BW) and
    // AVX-512 Doubleword and Quadword Instructions (AVX-512DQ) support:
    // Both are indicated by bits in EBX from leaf 7, sub-leaf 0:
    // AVX-512BW by bit 30 and AVX-512DQ by bit 17.
    avx512bw_supported = ebx & (1 << 30);
    avx512dq_supported = ebx & (1 << 17);

    // Check for AVX-512 Vector Length Extensions (AVX-512VL) support:
    // It is indicated by bit 31 of EBX from leaf 7, sub-leaf 0.
    avx512vl_supported = ebx & (1 << 31);

    return avx512f_supported && avx512cd_supported && avx512bw_supported && avx512dq_supported && avx512vl_supported;
}
