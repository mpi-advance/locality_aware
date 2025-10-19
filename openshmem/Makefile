CC ?= oshcc
CFLAGS ?= -O3 -std=c99 -Iinclude

all: ex_persistent bench_alltoallv

lib: src/osh_alltoallv.c
	$(CC) $(CFLAGS) -c src/osh_alltoallv.c -o osh_alltoallv.o
	ar rcs libosh_a2av.a osh_alltoallv.o

ex_persistent: lib examples/ex_persistent_alltoallv.c
	$(CC) $(CFLAGS) examples/ex_persistent_alltoallv.c libosh_a2av.a -o ex_persistent

bench_alltoallv: lib bench/bench_alltoallv.c
	$(CC) $(CFLAGS) bench/bench_alltoallv.c libosh_a2av.a -o bench_alltoallv

clean:
	rm -f *.o *.a ex_persistent bench_alltoallv
