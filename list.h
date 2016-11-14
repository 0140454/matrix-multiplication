#ifndef __ALGO_LIST_H
#define __ALGO_LIST_H

#define FUNC_REGISTER(name) \
		algo_t name = { .type = #name, .join = name ## _multiply, .pNext = NULL }; \
		__attribute__((constructor)) void append_##name() { \
			if (list == NULL) { list = &name; } \
			else { \
				algo_t *tmp; \
				for (tmp = list; tmp->pNext != NULL; tmp = tmp->pNext); \
				tmp->pNext = &name; \
			} \
		}

#define FUNC_BEGIN(name) \
	void name ## _multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h, int src2_w, int src2_h) {

#if defined(benchmark)

#define FUNC_END(name) } \
		FUNC_REGISTER(name)

#else

#define FUNC_END(name) }

#endif

typedef struct __ALGORITHM {
    char *type;
    void (*join) (int *src1, int *src2, int *dst, int src1_w, int src1_h, int src2_w, int src2_h);
    struct __ALGORITHM *pNext;
} algo_t;

#endif
