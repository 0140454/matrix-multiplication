#ifndef __ALGO_LIST_H
#define __ALGO_LIST_H

#define FUNC_REGISTER(name) \
		__attribute__((constructor)) void append_##name() { \
			static algo_t name = { .type = #name, .join = name ## _multiply, .pNext = NULL }; \
			if (list == NULL) { list = &name; } \
			else { \
				algo_t *tmp; \
				for (tmp = list; tmp->pNext != NULL; tmp = tmp->pNext); \
				tmp->pNext = &name; \
			} \
		}

#if defined(benchmark)

#define FUNC_IMPL(name) \
		void name ## _multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h, int src2_w, int src2_h); \
		FUNC_REGISTER(name); \
		void name ## _multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h, int src2_w, int src2_h)

#else

#define FUNC_IMPL(name) \
		void name ## _multiply(int *src1, int *src2, int *dst, int src1_w, int src1_h, int src2_w, int src2_h)

#endif

typedef struct __ALGORITHM {
    char *type;
    void (*join) (int *src1, int *src2, int *dst, int src1_w, int src1_h,
                  int src2_w, int src2_h);
    struct __ALGORITHM *pNext;
} algo_t;

#endif
