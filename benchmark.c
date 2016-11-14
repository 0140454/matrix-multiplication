#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#include <immintrin.h>
#include <malloc.h>

#include "list.h"
#include "impl.c"

#define TEST_W 1024
#define TEST_H 1024

algo_t *list;

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
    struct timespec start, end;
    int *src1 = (int *) memalign(32, sizeof(int) * TEST_W * TEST_H);
    int *src2 = (int *) memalign(32, sizeof(int) * TEST_W * TEST_H);
    int *dst = (int *) memalign(32, sizeof(int) * TEST_W * TEST_H);

    srand(time(NULL));
    for (int i = 0; i < TEST_H; ++i) {
        for (int j = 0; j < TEST_W; ++j) {
            src1[i * TEST_W + j] = rand();
            src2[i * TEST_W + j] = rand();
        }
    }

    algo_t *tmp = list;
    for (tmp = list; tmp != NULL; tmp = tmp->pNext) {
        clock_gettime(CLOCK_REALTIME, &start);
        tmp->join(src1, src2, dst, TEST_W, TEST_H, TEST_W, TEST_H);
        clock_gettime(CLOCK_REALTIME, &end);
        printf("%s %ld\n", tmp->type, diff_in_us(start, end));
    }


    return 0;
}
