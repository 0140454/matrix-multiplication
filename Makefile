CFLAGS = -mavx2 --std gnu99 -O0 -Wall

GIT_HOOKS := .git/hooks/pre-commit

VERIFY ?= 0
COMMON_SRCS := main.c impl.c
EXECUTABLE := naive submatrix sse sse_prefetch avx avx_prefetch

all: $(GIT_HOOKS) $(EXECUTABLE)

naive: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

submatrix: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

sse: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

sse_prefetch: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

avx: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

avx_prefetch: $(COMMON_SRCS)
	$(CC) $(CFLAGS) -DVERIFY=$(VERIFY) -D$@ -o $@ main.c

cache-test: all
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./naive
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./submatrix
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./sse
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./sse_prefetch
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./avx
	echo 1 | sudo tee /proc/sys/vm/drop_caches && perf stat --repeat 10 -e cache-misses,cache-references,instructions,cycles ./avx_prefetch

test: all
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./naive
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./submatrix
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./sse
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./sse_prefetch
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./avx
	echo 1 | sudo tee /proc/sys/vm/drop_caches && ./avx_prefetch

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

clean:
	$(RM) $(EXECUTABLE)
