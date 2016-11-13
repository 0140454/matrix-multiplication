#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <immintrin.h>
#include <malloc.h>

#include "list.h"
#include "impl.c"

algo_t *list;

int main(int argc, char *argv[])
{
    //TODO
    algo_t *tmp = list;
    for (tmp = list; tmp != NULL; tmp = tmp->pNext) {
        printf("%s\n", tmp->type);
    }


    return 0;
}
