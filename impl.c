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

int *strassen_add(int *src1, int *src2, int *dst, int size)
{
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            dst[i * size + k] = src1[i * size + k] + src2[i * size + k];
        }
    }

    return dst;
}

int *strassen_minus(int *src1, int *src2, int *dst, int size)
{
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            dst[i * size + k] = src1[i * size + k] - src2[i * size + k];
        }
    }

    return dst;
}

void strassen_multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h,
                       int src2_w, int src2_h)
{
    if (src1_w <= 2) {
        dst[0] = src1[0] * src2[0] + src1[1] * src2[2];
        dst[1] = src1[0] * src2[1] + src1[1] * src2[3];
        dst[2] = src1[2] * src2[0] + src1[3] * src2[2];
        dst[3] = src1[2] * src2[1] + src1[3] * src2[3];

        return;
    }

    int new_size = src1_w / 2;
    int *buffer = (int *) malloc(sizeof(int) * new_size * new_size * 21);
    int *a11 = buffer, *a12 = buffer + new_size * new_size,
         *a21 = buffer + new_size * new_size * 2,
          *a22 = buffer + new_size * new_size * 3,
           *b11 = buffer + new_size * new_size * 4,
            *b12 = buffer + new_size * new_size * 5,
             *b21 = buffer + new_size * new_size * 6,
              *b22 = buffer + new_size * new_size * 7,
               *c11 = buffer + new_size * new_size * 8,
                *c12 = buffer + new_size * new_size * 9,
                 *c21 = buffer + new_size * new_size * 10,
                  *c22 = buffer + new_size * new_size * 11;
    int *m1 = buffer + new_size * new_size * 12,
         *m2 = buffer + new_size * new_size * 13,
          *m3 = buffer + new_size * new_size * 14,
           *m4 = buffer + new_size * new_size * 15,
            *m5 = buffer + new_size * new_size * 16,
             *m6 = buffer + new_size * new_size * 17,
              *m7 = buffer + new_size * new_size * 18;
    int *tmp1 = buffer + new_size * new_size * 19,
         *tmp2 = buffer + new_size * new_size * 20;

    for (int i = 0; i < new_size; ++i) {
        memcpy(a11 + i * new_size, src1 + i * src1_w, sizeof(int) * new_size);
        memcpy(a12 + i * new_size, src1 + i * src1_w + new_size,
               sizeof(int) * new_size);
        memcpy(a21 + i * new_size, src1 + (i + new_size) * src1_w,
               sizeof(int) * new_size);
        memcpy(a22 + i * new_size, src1 + (i + new_size) * src1_w + new_size,
               sizeof(int) * new_size);

        memcpy(b11 + i * new_size, src2 + i * src2_w, sizeof(int) * new_size);
        memcpy(b12 + i * new_size, src2 + i * src2_w + new_size,
               sizeof(int) * new_size);
        memcpy(b21 + i * new_size, src2 + (i + new_size) * src2_w,
               sizeof(int) * new_size);
        memcpy(b22 + i * new_size, src2 + (i + new_size) * src2_w + new_size,
               sizeof(int) * new_size);
    }

    strassen_multiply(strassen_add(a11, a22, tmp1, new_size),
                      strassen_add(b11, b22, tmp2, new_size),
                      m1, new_size, new_size, new_size, new_size);

    strassen_multiply(strassen_add(a21, a22, tmp1, new_size),
                      b11,
                      m2, new_size, new_size, new_size, new_size);

    strassen_multiply(a11,
                      strassen_minus(b12, b22, tmp2, new_size),
                      m3, new_size, new_size, new_size, new_size);

    strassen_multiply(a22,
                      strassen_minus(b21, b11, tmp2, new_size),
                      m4, new_size, new_size, new_size, new_size);

    strassen_multiply(strassen_add(a11, a12, tmp1, new_size),
                      b22,
                      m5, new_size, new_size, new_size, new_size);

    strassen_multiply(strassen_minus(a21, a11, tmp1, new_size),
                      strassen_add(b11, b12, tmp2, new_size),
                      m6, new_size, new_size, new_size, new_size);

    strassen_multiply(strassen_minus(a12, a22, tmp1, new_size),
                      strassen_add(b21, b22, tmp2, new_size),
                      m7, new_size, new_size, new_size, new_size);

    strassen_add(m1, m4, tmp1, new_size);
    strassen_minus(m5, m7, tmp2, new_size);
    strassen_minus(tmp1, tmp2, c11, new_size);

    strassen_add(m3, m5, c12, new_size);
    strassen_add(m2, m4, c21, new_size);

    strassen_minus(m1, m2, tmp1, new_size);
    strassen_add(m3, m6, tmp2, new_size);
    strassen_add(tmp1, tmp2, c22, new_size);

    for (int i = 0; i < new_size; ++i) {
        memcpy(dst + i * src1_w, c11 + i * new_size, sizeof(int) * new_size);
        memcpy(dst + i * src1_w + new_size, c12 + i * new_size, sizeof(int) * new_size);
        memcpy(dst + (i + new_size) * src1_w, c21 + i * new_size,
               sizeof(int) * new_size);
        memcpy(dst + (i + new_size) * src1_w + new_size, c22 + i * new_size,
               sizeof(int) * new_size);
    }

    free(buffer);
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

#endif
