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
}

void sse_prefetch_multiply(int *src1, int *src2, int *dst, int src1_w,
                           int src1_h, int src2_w, int src2_h)
{
}

#endif
