#ifndef _MUTIPLY_H
#define _MULTIPLY_H
int *add_s(int *res,int *a, int *b, int n);
int *sub_s(int *res,int *x, int *y, int n);
int *mul_s(int *dst,int *a, int *b, int n);
void concer(int*dst, int *m11, int *m12, int *m21, int *m22, int n);
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
            __m128i des0 = _mm_setzero_si128 ();
            __m128i des1 = _mm_setzero_si128 ();
            __m128i des2 = _mm_setzero_si128 ();
            __m128i des3 = _mm_setzero_si128 ();

            for (int k = 0; k < src2_w; k += 4) {
                __m128i I0 = _mm_load_si128((__m128i *)(src1 + (x + 0) * src1_w + k));
                __m128i I1 = _mm_load_si128((__m128i *)(src1 + (x + 1) * src1_w + k));
                __m128i I2 = _mm_load_si128((__m128i *)(src1 + (x + 2) * src1_w + k));
                __m128i I3 = _mm_load_si128((__m128i *)(src1 + (x + 3) * src1_w + k));

                __m128i I4 = _mm_set_epi32 (src2[(k+3) * src2_w + y], src2[(k+2) * src2_w + y],
                                            src2[(k+1) * src2_w + y], src2[k * src2_w + y]);
                __m128i I5 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+1)],
                                            src2[(k+2) * src2_w + (y+1)], src2[(k+1) * src2_w + (y+1)],
                                            src2[(k+0) * src2_w + (y+1)]);
                __m128i I6 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+2)],
                                            src2[(k+2) * src2_w + (y+2)], src2[(k+1) * src2_w + (y+2)],
                                            src2[(k+0) * src2_w + (y+2)]);
                __m128i I7 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+3)],
                                            src2[(k+2) * src2_w + (y+3)], src2[(k+1) * src2_w + (y+3)],
                                            src2[(k+0) * src2_w + (y+3)]);

                __m128i T0 = _mm_mullo_epi32(I0, I4);
                __m128i T1 = _mm_mullo_epi32(I0, I5);
                __m128i T2 = _mm_mullo_epi32(I0, I6);
                __m128i T3 = _mm_mullo_epi32(I0, I7);

                __m128i T4 = _mm_mullo_epi32(I1, I4);
                __m128i T5 = _mm_mullo_epi32(I1, I5);
                __m128i T6 = _mm_mullo_epi32(I1, I6);
                __m128i T7 = _mm_mullo_epi32(I1, I7);

                __m128i T8 = _mm_mullo_epi32(I2, I4);
                __m128i T9 = _mm_mullo_epi32(I2, I5);
                __m128i T10 = _mm_mullo_epi32(I2, I6);
                __m128i T11 = _mm_mullo_epi32(I2, I7);

                __m128i T12 = _mm_mullo_epi32(I3, I4);
                __m128i T13 = _mm_mullo_epi32(I3, I5);
                __m128i T14 = _mm_mullo_epi32(I3, I6);
                __m128i T15 = _mm_mullo_epi32(I3, I7);

                __m128i T16 = _mm_unpacklo_epi32(T0, T1);
                __m128i T17 = _mm_unpacklo_epi32(T2, T3);
                __m128i T18 = _mm_unpackhi_epi32(T0, T1);
                __m128i T19 = _mm_unpackhi_epi32(T2, T3);

                __m128i T20 = _mm_unpacklo_epi64(T16, T17);
                __m128i T21 = _mm_unpackhi_epi64(T16, T17);
                __m128i T22 = _mm_unpacklo_epi64(T18, T19);
                __m128i T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des0 = _mm_add_epi32(T20, des0);

                T16 = _mm_unpacklo_epi32(T4, T5);
                T17 = _mm_unpacklo_epi32(T6, T7);
                T18 = _mm_unpackhi_epi32(T4, T5);
                T19 = _mm_unpackhi_epi32(T6, T7);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des1 = _mm_add_epi32(T20, des1);

                T16 = _mm_unpacklo_epi32(T8, T9);
                T17 = _mm_unpacklo_epi32(T10, T11);
                T18 = _mm_unpackhi_epi32(T8, T9);
                T19 = _mm_unpackhi_epi32(T10, T11);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des2 = _mm_add_epi32(T20, des2);

                T16 = _mm_unpacklo_epi32(T12, T13);
                T17 = _mm_unpacklo_epi32(T14, T15);
                T18 = _mm_unpackhi_epi32(T12, T13);
                T19 = _mm_unpackhi_epi32(T14, T15);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des3 = _mm_add_epi32(T20, des3);
            }

            _mm_store_si128((__m128i *)(dst + ((x + 0) * src2_w) + y), des0);
            _mm_store_si128((__m128i *)(dst + ((x + 1) * src2_w) + y), des1);
            _mm_store_si128((__m128i *)(dst + ((x + 2) * src2_w) + y), des2);
            _mm_store_si128((__m128i *)(dst + ((x + 3) * src2_w) + y), des3);
        }
    }
}

void sse_prefetch_multiply(int *src1, int *src2, int *dst, int src1_w,
                           int src1_h, int src2_w, int src2_h)
{
    for (int x = 0; x < src1_h; x += 4) {
        for (int y = 0; y < src2_w; y += 4) {
            __m128i des0 = _mm_setzero_si128 ();
            __m128i des1 = _mm_setzero_si128 ();
            __m128i des2 = _mm_setzero_si128 ();
            __m128i des3 = _mm_setzero_si128 ();

            for (int k = 0; k < src2_w; k += 4) {
#define SSE_PFDIST  8
                _mm_prefetch(src2 + (k + SSE_PFDIST + 0) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 1) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 2) * src2_w + y, _MM_HINT_T1);
                _mm_prefetch(src2 + (k + SSE_PFDIST + 3) * src2_w + y, _MM_HINT_T1);

                __m128i I0 = _mm_load_si128((__m128i *)(src1 + (x + 0) * src1_w + k));
                __m128i I1 = _mm_load_si128((__m128i *)(src1 + (x + 1) * src1_w + k));
                __m128i I2 = _mm_load_si128((__m128i *)(src1 + (x + 2) * src1_w + k));
                __m128i I3 = _mm_load_si128((__m128i *)(src1 + (x + 3) * src1_w + k));

                __m128i I4 = _mm_set_epi32 (src2[(k+3) * src2_w + y], src2[(k+2) * src2_w + y],
                                            src2[(k+1) * src2_w + y], src2[k * src2_w + y]);
                __m128i I5 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+1)],
                                            src2[(k+2) * src2_w + (y+1)], src2[(k+1) * src2_w + (y+1)],
                                            src2[(k+0) * src2_w + (y+1)]);
                __m128i I6 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+2)],
                                            src2[(k+2) * src2_w + (y+2)], src2[(k+1) * src2_w + (y+2)],
                                            src2[(k+0) * src2_w + (y+2)]);
                __m128i I7 = _mm_set_epi32 (src2[(k+3) * src2_w + (y+3)],
                                            src2[(k+2) * src2_w + (y+3)], src2[(k+1) * src2_w + (y+3)],
                                            src2[(k+0) * src2_w + (y+3)]);

                __m128i T0 = _mm_mullo_epi32(I0, I4);
                __m128i T1 = _mm_mullo_epi32(I0, I5);
                __m128i T2 = _mm_mullo_epi32(I0, I6);
                __m128i T3 = _mm_mullo_epi32(I0, I7);

                __m128i T4 = _mm_mullo_epi32(I1, I4);
                __m128i T5 = _mm_mullo_epi32(I1, I5);
                __m128i T6 = _mm_mullo_epi32(I1, I6);
                __m128i T7 = _mm_mullo_epi32(I1, I7);

                __m128i T8 = _mm_mullo_epi32(I2, I4);
                __m128i T9 = _mm_mullo_epi32(I2, I5);
                __m128i T10 = _mm_mullo_epi32(I2, I6);
                __m128i T11 = _mm_mullo_epi32(I2, I7);

                __m128i T12 = _mm_mullo_epi32(I3, I4);
                __m128i T13 = _mm_mullo_epi32(I3, I5);
                __m128i T14 = _mm_mullo_epi32(I3, I6);
                __m128i T15 = _mm_mullo_epi32(I3, I7);

                __m128i T16 = _mm_unpacklo_epi32(T0, T1);
                __m128i T17 = _mm_unpacklo_epi32(T2, T3);
                __m128i T18 = _mm_unpackhi_epi32(T0, T1);
                __m128i T19 = _mm_unpackhi_epi32(T2, T3);

                __m128i T20 = _mm_unpacklo_epi64(T16, T17);
                __m128i T21 = _mm_unpackhi_epi64(T16, T17);
                __m128i T22 = _mm_unpacklo_epi64(T18, T19);
                __m128i T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des0 = _mm_add_epi32(T20, des0);

                T16 = _mm_unpacklo_epi32(T4, T5);
                T17 = _mm_unpacklo_epi32(T6, T7);
                T18 = _mm_unpackhi_epi32(T4, T5);
                T19 = _mm_unpackhi_epi32(T6, T7);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des1 = _mm_add_epi32(T20, des1);

                T16 = _mm_unpacklo_epi32(T8, T9);
                T17 = _mm_unpacklo_epi32(T10, T11);
                T18 = _mm_unpackhi_epi32(T8, T9);
                T19 = _mm_unpackhi_epi32(T10, T11);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des2 = _mm_add_epi32(T20, des2);

                T16 = _mm_unpacklo_epi32(T12, T13);
                T17 = _mm_unpacklo_epi32(T14, T15);
                T18 = _mm_unpackhi_epi32(T12, T13);
                T19 = _mm_unpackhi_epi32(T14, T15);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des3 = _mm_add_epi32(T20, des3);
            }

            _mm_store_si128((__m128i *)(dst + ((x + 0) * src2_w) + y), des0);
            _mm_store_si128((__m128i *)(dst + ((x + 1) * src2_w) + y), des1);
            _mm_store_si128((__m128i *)(dst + ((x + 2) * src2_w) + y), des2);
            _mm_store_si128((__m128i *)(dst + ((x + 3) * src2_w) + y), des3);
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
                ymm0 = _mm256_load_si256((__m256i *) (src2 + (k + 0) * src2_w + j));
                ymm1 = _mm256_load_si256((__m256i *) (src2 + (k + 1) * src2_w + j));
                ymm2 = _mm256_load_si256((__m256i *) (src2 + (k + 2) * src2_w + j));
                ymm3 = _mm256_load_si256((__m256i *) (src2 + (k + 3) * src2_w + j));
                ymm4 = _mm256_load_si256((__m256i *) (src2 + (k + 4) * src2_w + j));
                ymm5 = _mm256_load_si256((__m256i *) (src2 + (k + 5) * src2_w + j));
                ymm6 = _mm256_load_si256((__m256i *) (src2 + (k + 6) * src2_w + j));
                ymm7 = _mm256_load_si256((__m256i *) (src2 + (k + 7) * src2_w + j));

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

            _mm256_store_si256((__m256i *) (dst + (i + 0) * src2_w + j), ymm16);
            _mm256_store_si256((__m256i *) (dst + (i + 1) * src2_w + j), ymm17);
            _mm256_store_si256((__m256i *) (dst + (i + 2) * src2_w + j), ymm18);
            _mm256_store_si256((__m256i *) (dst + (i + 3) * src2_w + j), ymm19);
            _mm256_store_si256((__m256i *) (dst + (i + 4) * src2_w + j), ymm20);
            _mm256_store_si256((__m256i *) (dst + (i + 5) * src2_w + j), ymm21);
            _mm256_store_si256((__m256i *) (dst + (i + 6) * src2_w + j), ymm22);
            _mm256_store_si256((__m256i *) (dst + (i + 7) * src2_w + j), ymm23);
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
                ymm0 = _mm256_load_si256((__m256i *) (src2 + (k + 0) * src2_w + j));
                ymm1 = _mm256_load_si256((__m256i *) (src2 + (k + 1) * src2_w + j));
                ymm2 = _mm256_load_si256((__m256i *) (src2 + (k + 2) * src2_w + j));
                ymm3 = _mm256_load_si256((__m256i *) (src2 + (k + 3) * src2_w + j));
                ymm4 = _mm256_load_si256((__m256i *) (src2 + (k + 4) * src2_w + j));
                ymm5 = _mm256_load_si256((__m256i *) (src2 + (k + 5) * src2_w + j));
                ymm6 = _mm256_load_si256((__m256i *) (src2 + (k + 6) * src2_w + j));
                ymm7 = _mm256_load_si256((__m256i *) (src2 + (k + 7) * src2_w + j));

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

            _mm256_store_si256((__m256i *) (dst + (i + 0) * src2_w + j), ymm16);
            _mm256_store_si256((__m256i *) (dst + (i + 1) * src2_w + j), ymm17);
            _mm256_store_si256((__m256i *) (dst + (i + 2) * src2_w + j), ymm18);
            _mm256_store_si256((__m256i *) (dst + (i + 3) * src2_w + j), ymm19);
            _mm256_store_si256((__m256i *) (dst + (i + 4) * src2_w + j), ymm20);
            _mm256_store_si256((__m256i *) (dst + (i + 5) * src2_w + j), ymm21);
            _mm256_store_si256((__m256i *) (dst + (i + 6) * src2_w + j), ymm22);
            _mm256_store_si256((__m256i *) (dst + (i + 7) * src2_w + j), ymm23);
        }
    }
}

int *Strassen(int *res,int *src1, int *src2, int n)
{


    int n2= n/2;
    int nn = n2*n2;
    int *buffer = (int*) malloc(sizeof(int)*nn*21);
    int *a11 = buffer,
         *a12 = buffer + nn,
          *a21 = buffer + nn * 2,
           *a22 = buffer + nn * 3,
            *b11 = buffer + nn * 4,
             *b12 = buffer + nn * 5,
              *b21 = buffer + nn * 6,
               *b22 = buffer + nn * 7,
                *m1 = buffer + nn * 8,
                 *m2 = buffer + nn * 9,
                  *m3 = buffer + nn * 10,
                   *m4 = buffer + nn * 11;

    int *p1 = buffer + nn * 12,
         *p2 = buffer + nn * 13,
          *p3 = buffer + nn * 14,
           *p4 = buffer + nn * 15,
            *p5 = buffer + nn * 16,
             *p6 = buffer + nn * 17,
              *p7 = buffer + nn * 18;
    int *ares = buffer + nn * 19,
         *bres = buffer + nn * 20;


    for(int i=0; i<n2; i++) {
        for(int j=0; j<n2; j++) {
            a11[i*n2+j] = src1[i*n+j];
            a12[i*n2+j] = src1[i*n+j+n2];
            a21[i*n2+j] = src1[(i+n2)*n+j];
            a22[i*n2+j] = src1[(i+n2)*n+j+(n2)];

            b11[i*n2+j] = src2[(i*n)+j];
            b12[i*n2+j] = src2[(i*n)+j+n2];
            b21[i*n2+j] = src2[(i+n2)*n+j];
            b22[i*n2+j] = src2[(i+n2)*n+j+(n2)];

        }

    }


    mul_s(p1,sub_s(ares,a12,a22,n2),add_s(bres,b21,b22,n2),n2);
    mul_s(p2,add_s(ares,a11,a22,n2),add_s(bres,b11,b22,n2),n2);
    mul_s(p3,sub_s(ares,a11,a21,n2),add_s(bres,b11,b12,n2),n2);
    mul_s(p4,add_s(ares,a11,a12,n2),b22,n2);
    mul_s(p5,a11,sub_s(bres,b12,b22,n2),n2);
    mul_s(p6,a22,sub_s(bres,b21,b11,n2),n2);
    mul_s(p7,add_s(ares,a21,a22,n2),b11,n2);


    sub_s(m1,add_s(ares,p6,add_s(bres,p1,p2,n2),n2),p4,n2);
    add_s(m2,p4,p5,n2);
    add_s(m3,p6,p7,n2);
    sub_s(m4,sub_s(ares,add_s(bres,p2,p5,n2),p7,n2),p3,n2);

    concer(res,m1,m2,m3,m4,n2);

    free(buffer);

    return res;
}

void concer(int*res, int *m11, int *m12, int *m21, int *m22,int n)
{
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            res[i*2*n+j] = m11[i*n+j];
            res[i*2*n+j+n] = m12[i*n+j];
            res[(i+n)*n*2+j] = m21[i*n+j];
            res[(i+n)*n*2+j+n] = m22[i*n+j];

        }

    }

}

int *add_s(int *res, int *x, int *y, int n)
{

    //  int *res= (int*) malloc(sizeof(int)*n*n);
    memset(res, 0, sizeof(int) * n * n);
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++)
            res[i*n+j] = x[i*n+j] + y[i*n+j];
    }
    return res;

}
int *sub_s(int *res,int *x, int *y, int n)
{

    //int *res= (int*) malloc(sizeof(int)*n*n);
    memset(res, 0, sizeof(int) * n * n);
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++)
            res[i*n+j] = x[i*n+j] - y[i*n+j];
    }
    return res;

}

int* mul_s(int *dst, int *a, int *b, int n)
{


    if(n == 2) {


        int P1 = (a[0*n+1]-a[1*n+1])*(b[1*n+0]+b[1*n+1]);
        int P2 = (a[0*n+0]+a[1*n+1])*(b[0*n+0]+b[1*n+1]);
        int P3 = (a[0*n+0]-a[1*n+0])*(b[0*n+0]+b[0*n+1]);
        int P4 = (a[0*n+0]+a[0*n+1])*b[1*n+1];
        int P5 = a[0*n+0]*(b[0*n+1]-b[1*n+1]);
        int P6 = a[1*n+1]*(b[1*n+0]-b[0*n+0]);
        int P7 = (a[1*n+0]+a[1*n+1])*b[0*n+0];

        dst[0*n+0] = P1 + P2 - P4 + P6;
        dst[0*n+1] = P4 + P5;
        dst[1*n+0] = P6 + P7;
        dst[1*n+1] = P2 - P3 + P5 - P7;
        return dst;
    }

    else
        return Strassen(dst,a,b,n);


}

void Strassen_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                       int src2_w, int src2_h)
{

    memset(dst, 0, sizeof(int) * src1_h * src2_w);


    Strassen(dst,src1,src2,src1_w);


}



#endif
