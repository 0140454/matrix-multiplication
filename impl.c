#ifndef _MUTIPLY_H
#define _MULTIPLY_H

void naive_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                    int src2_w, int src2_h)
{
    memset(dst, 0, sizeof(int) * src1_h * src2_w);

    for (int i = 0; i < src1_h; ++i) {
        for (int j = 0; j < src2_w; ++j) {
            for (int k = 0; k < src2_h; ++k) {
                dst[i * src2_w + j] += src1[i * src1_w + k] * src2[k * src2_w + j];
            }
        }
    }
}

void submatrix_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                        int src2_w, int src2_h)
{
    memset(dst, 0, sizeof(int) * src1_h * src2_w);

    for (int i = 0; i < src1_h; i += 4) {
        for (int j = 0; j < src2_w; j += 4) {
            for (int k = 0; k < src2_h; k += 4) {
                for (int i2 = 0; i2 < 4; ++i2) {
                    for (int j2 = 0; j2 < 4; ++j2) {
                        for (int k2 = 0; k2 < 4; ++k2) {
                            dst[(i + i2) * src2_w + (j + j2)] += src1[(i + i2) * src1_w + (k + k2)] *
                                                                 src2[(k + k2) * src2_w + (j + j2)];
                        }
                    }
                }
            }
        }
    }
}

void sse_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                  int src2_w, int src2_h)
{
    for (int x = 0; x < src1_h; x += 4) {
        for (int y = 0; y < src2_w; y += 4) {
            __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            __m128i xmm9 = _mm_setzero_si128 ();
            __m128i xmm10 = _mm_setzero_si128 ();
            __m128i xmm11 = _mm_setzero_si128 ();
            __m128i xmm12 = _mm_setzero_si128 ();

            for (int k = 0; k < src2_w; k += 4) {
                // load eight rows from source 2
                xmm0 = _mm_loadu_si128((__m128i *) (src2 + (k + 0) * src2_w + y));
                xmm1 = _mm_loadu_si128((__m128i *) (src2 + (k + 1) * src2_w + y));
                xmm2 = _mm_loadu_si128((__m128i *) (src2 + (k + 2) * src2_w + y));
                xmm3 = _mm_loadu_si128((__m128i *) (src2 + (k + 3) * src2_w + y));

                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 0) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm9 = _mm_add_epi32(xmm9, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 1) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm10 = _mm_add_epi32(xmm10, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 2) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm11 = _mm_add_epi32(xmm11, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 3) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm12 = _mm_add_epi32(xmm12, xmm5);
            }

            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * src2_w) + y), xmm9);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * src2_w) + y), xmm10);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * src2_w) + y), xmm11);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * src2_w) + y), xmm12);
        }
    }
}

void sse_prefetch_multiply(int *src1, int *src2, int *dst, int src1_w,
                           int src1_h, int src2_w, int src2_h)
{
    for (int x = 0; x < src1_h; x += 4) {
        for (int y = 0; y < src2_w; y += 4) {
            __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

            __m128i xmm9 = _mm_setzero_si128 ();
            __m128i xmm10 = _mm_setzero_si128 ();
            __m128i xmm11 = _mm_setzero_si128 ();
            __m128i xmm12 = _mm_setzero_si128 ();

            for (int k = 0; k < src2_w; k += 4) {
#define SSE_PFDIST  8
                _mm_prefetch(src2 + (k + SSE_PFDIST + 0) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 1) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 2) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 3) * src2_w + y, _MM_HINT_T1);

                // load eight rows from source 2
                xmm0 = _mm_loadu_si128((__m128i *) (src2 + (k + 0) * src2_w + y));
                xmm1 = _mm_loadu_si128((__m128i *) (src2 + (k + 1) * src2_w + y));
                xmm2 = _mm_loadu_si128((__m128i *) (src2 + (k + 2) * src2_w + y));
                xmm3 = _mm_loadu_si128((__m128i *) (src2 + (k + 3) * src2_w + y));

                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 0) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm9 = _mm_add_epi32(xmm9, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 1) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm10 = _mm_add_epi32(xmm10, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 2) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm11 = _mm_add_epi32(xmm11, xmm5);

                //------------------------------------------------------------//
                // broadcast each elements from source 1
                xmm4 = _mm_loadu_si128((__m128i *) (src1 + (x + 3) * src1_w + k));
                xmm5 = _mm_shuffle_epi32(xmm4, 0x00);
                xmm6 = _mm_shuffle_epi32(xmm4, 0x55);
                xmm7 = _mm_shuffle_epi32(xmm4, 0xAA);
                xmm8 = _mm_shuffle_epi32(xmm4, 0xFF);

                // multiply
                xmm5 = _mm_mullo_epi32(xmm5, xmm0); // row 1, 2
                xmm6 = _mm_mullo_epi32(xmm6, xmm1);
                xmm5 = _mm_add_epi32(xmm5, xmm6);

                xmm7 = _mm_mullo_epi32(xmm7, xmm2); // row 3, 4
                xmm8 = _mm_mullo_epi32(xmm8, xmm3);
                xmm7 = _mm_add_epi32(xmm7, xmm8);

                xmm5 = _mm_add_epi32(xmm5, xmm7); //sum

                // save current result
                xmm12 = _mm_add_epi32(xmm12, xmm5);
            }

            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * src2_w) + y), xmm9);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * src2_w) + y), xmm10);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * src2_w) + y), xmm11);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * src2_w) + y), xmm12);
        }
    }
}

void avx_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                  int src2_w, int src2_h)
{
    for (int i = 0; i < src1_h; i += 8) {
        for (int j = 0; j < src2_w; j += 8) {
            __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

            __m256i ymm16 = _mm256_setzero_si256();
            __m256i ymm17 = _mm256_setzero_si256();
            __m256i ymm18 = _mm256_setzero_si256();
            __m256i ymm19 = _mm256_setzero_si256();
            __m256i ymm20 = _mm256_setzero_si256();
            __m256i ymm21 = _mm256_setzero_si256();
            __m256i ymm22 = _mm256_setzero_si256();
            __m256i ymm23 = _mm256_setzero_si256();

            for (int k = 0; k < src2_h; k += 8) {
                // load eight rows from source 2
                ymm0 = _mm256_loadu_si256((__m256i *) (src2 + (k + 0) * src2_w + j));
                ymm1 = _mm256_loadu_si256((__m256i *) (src2 + (k + 1) * src2_w + j));
                ymm2 = _mm256_loadu_si256((__m256i *) (src2 + (k + 2) * src2_w + j));
                ymm3 = _mm256_loadu_si256((__m256i *) (src2 + (k + 3) * src2_w + j));
                ymm4 = _mm256_loadu_si256((__m256i *) (src2 + (k + 4) * src2_w + j));
                ymm5 = _mm256_loadu_si256((__m256i *) (src2 + (k + 5) * src2_w + j));
                ymm6 = _mm256_loadu_si256((__m256i *) (src2 + (k + 6) * src2_w + j));
                ymm7 = _mm256_loadu_si256((__m256i *) (src2 + (k + 7) * src2_w + j));

                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm16 = _mm256_add_epi32(ymm16, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm17 = _mm256_add_epi32(ymm17, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm18 = _mm256_add_epi32(ymm18, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm19 = _mm256_add_epi32(ymm19, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm20 = _mm256_add_epi32(ymm20, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm21 = _mm256_add_epi32(ymm21, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm22 = _mm256_add_epi32(ymm22, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm23 = _mm256_add_epi32(ymm23, ymm8);
            }

            _mm256_storeu_si256((__m256i *) (dst + (i + 0) * src2_w + j), ymm16);
            _mm256_storeu_si256((__m256i *) (dst + (i + 1) * src2_w + j), ymm17);
            _mm256_storeu_si256((__m256i *) (dst + (i + 2) * src2_w + j), ymm18);
            _mm256_storeu_si256((__m256i *) (dst + (i + 3) * src2_w + j), ymm19);
            _mm256_storeu_si256((__m256i *) (dst + (i + 4) * src2_w + j), ymm20);
            _mm256_storeu_si256((__m256i *) (dst + (i + 5) * src2_w + j), ymm21);
            _mm256_storeu_si256((__m256i *) (dst + (i + 6) * src2_w + j), ymm22);
            _mm256_storeu_si256((__m256i *) (dst + (i + 7) * src2_w + j), ymm23);
        }
    }
}

void avx_prefetch_multiply(int *src1, int *src2, int *dst, int src1_w,
                           int src1_h, int src2_w, int src2_h)
{
    for (int i = 0; i < src1_h; i += 8) {
        for (int j = 0; j < src2_w; j += 8) {
            __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

            __m256i ymm16 = _mm256_setzero_si256();
            __m256i ymm17 = _mm256_setzero_si256();
            __m256i ymm18 = _mm256_setzero_si256();
            __m256i ymm19 = _mm256_setzero_si256();
            __m256i ymm20 = _mm256_setzero_si256();
            __m256i ymm21 = _mm256_setzero_si256();
            __m256i ymm22 = _mm256_setzero_si256();
            __m256i ymm23 = _mm256_setzero_si256();

            for (int k = 0; k < src2_h; k += 8) {
#define AVX_PFDIST  8
                _mm_prefetch(src2 + (k + AVX_PFDIST + 0) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 1) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 2) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 3) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 4) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 5) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 6) * src2_w + j, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + AVX_PFDIST + 7) * src2_w + j, _MM_HINT_T1);

                // load eight rows from source 2
                ymm0 = _mm256_loadu_si256((__m256i *) (src2 + (k + 0) * src2_w + j));
                ymm1 = _mm256_loadu_si256((__m256i *) (src2 + (k + 1) * src2_w + j));
                ymm2 = _mm256_loadu_si256((__m256i *) (src2 + (k + 2) * src2_w + j));
                ymm3 = _mm256_loadu_si256((__m256i *) (src2 + (k + 3) * src2_w + j));
                ymm4 = _mm256_loadu_si256((__m256i *) (src2 + (k + 4) * src2_w + j));
                ymm5 = _mm256_loadu_si256((__m256i *) (src2 + (k + 5) * src2_w + j));
                ymm6 = _mm256_loadu_si256((__m256i *) (src2 + (k + 6) * src2_w + j));
                ymm7 = _mm256_loadu_si256((__m256i *) (src2 + (k + 7) * src2_w + j));

                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 0) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm16 = _mm256_add_epi32(ymm16, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 1) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm17 = _mm256_add_epi32(ymm17, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 2) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm18 = _mm256_add_epi32(ymm18, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 3) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm19 = _mm256_add_epi32(ymm19, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 4) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm20 = _mm256_add_epi32(ymm20, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 5) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm21 = _mm256_add_epi32(ymm21, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 6) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm22 = _mm256_add_epi32(ymm22, ymm8);

                // ---------------------------------------------------------- //
                // broadcast each elements from source 1
                ymm8 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 0]);
                ymm9 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 1]);
                ymm10 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 2]);
                ymm11 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 3]);
                ymm12 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 4]);
                ymm13 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 5]);
                ymm14 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 6]);
                ymm15 = _mm256_set1_epi32(src1[(i + 7) * src1_w + k + 7]);

                // multiply
                ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                ymm8 = _mm256_add_epi32(ymm8, ymm9);

                ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                ymm10 = _mm256_add_epi32(ymm10, ymm11);

                ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                ymm12 = _mm256_add_epi32(ymm12, ymm13);

                ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                ymm14 = _mm256_add_epi32(ymm14, ymm15);

                ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                ymm12 = _mm256_add_epi32(ymm12, ymm14);
                ymm8 = _mm256_add_epi32(ymm8, ymm12);

                // save current result
                ymm23 = _mm256_add_epi32(ymm23, ymm8);
            }

            _mm256_storeu_si256((__m256i *) (dst + (i + 0) * src2_w + j), ymm16);
            _mm256_storeu_si256((__m256i *) (dst + (i + 1) * src2_w + j), ymm17);
            _mm256_storeu_si256((__m256i *) (dst + (i + 2) * src2_w + j), ymm18);
            _mm256_storeu_si256((__m256i *) (dst + (i + 3) * src2_w + j), ymm19);
            _mm256_storeu_si256((__m256i *) (dst + (i + 4) * src2_w + j), ymm20);
            _mm256_storeu_si256((__m256i *) (dst + (i + 5) * src2_w + j), ymm21);
            _mm256_storeu_si256((__m256i *) (dst + (i + 6) * src2_w + j), ymm22);
            _mm256_storeu_si256((__m256i *) (dst + (i + 7) * src2_w + j), ymm23);
        }
    }
}

#endif
