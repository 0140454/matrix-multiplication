#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#include <xmmintrin.h>

#define TEST_W 4096
#define TEST_H 4096

#include "impl.c"

static long diff_in_us(struct timespec t1, struct timespec t2)
{
    struct timespec diff;
    if (t2.tv_nsec-t1.tv_nsec < 0) {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec - 1;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec + 1000000000;
    } else {
        diff.tv_sec  = t2.tv_sec - t1.tv_sec;
        diff.tv_nsec = t2.tv_nsec - t1.tv_nsec;
    }
    return (diff.tv_sec * 1000000.0 + diff.tv_nsec / 1000.0);
}

int main(int argc, char *argv[])
{
    /* verify the result of 4x4 matrix */
    {
        int test_src1[16] = { 0,  1,  2,  3,
                              4,  5,  6,  7,
                              8,  9, 10, 11,
                              12, 13, 14, 15
                            };
        int test_src2[16] = { 16, 17, 18, 19,
                              20, 21, 22, 23,
                              24, 25, 26, 27,
                              28, 29, 30, 31
                            };
        int testout[16];
        int expected[16] = { 152,  158,  164,  170,
                             504,  526,  548,  570,
                             856,  894,  932,  970,
                             1208, 1262, 1316, 1370
                           };

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++)
                printf(" %2d", test_src1[y * 4 + x]);
            printf("\n");
        }
        printf("\n");

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++)
                printf(" %2d", test_src2[y * 4 + x]);
            printf("\n");
        }
        printf("\n");

        naive_multiply(test_src1, test_src2, testout, 4, 4, 4, 4);

        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++)
                printf(" %2d", testout[y * 4 + x]);
            printf("\n");
        }

        assert(0 == memcmp(testout, expected, 16 * sizeof(int)) &&
               "Verification fails");
    }

    return 0;
}
