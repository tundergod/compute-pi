CC = gcc
CFLAGS = -O0 -std=gnu99 -Wall -fopenmp -mavx
EXECUTABLE = \
	time_test_baseline time_test_openmp_2 time_test_openmp_4 \
	time_test_avx time_test_avxunroll \
	time_test_leibniz time_test_leibniz_avx time_test_leibniz_avx_unroll \
	time_test_montecarlo benchmark_clock_gettime error \
	time_test_montecarlo_openmp_2 time_test_montecarlo_openmp_4

default: computepi.o
	$(CC) $(CFLAGS) computepi.o time_test.c -DBASELINE -o time_test_baseline
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_2 -o time_test_openmp_2
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_4 -o time_test_openmp_4
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVX -o time_test_avx
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVXUNROLL -o time_test_avxunroll
	$(CC) $(CFLAGS) computepi.o time_test.c -DLB -o time_test_leibniz
	$(CC) $(CFLAGS) computepi.o time_test.c -DLBAVX -o time_test_leibniz_avx
	$(CC) $(CFLAGS) computepi.o time_test.c -DLBAVXUNROLL -o time_test_leibniz_avx_unroll
	$(CC) $(CFLAGS) computepi.o time_test.c -DMC -o time_test_montecarlo
	$(CC) $(CFLAGS) computepi.o time_test.c -DMC_OPENMP_2 -o time_test_montecarlo_openmp_2
	$(CC) $(CFLAGS) computepi.o time_test.c -DMC_OPENMP_4 -o time_test_montecarlo_openmp_4
	$(CC) $(CFLAGS) computepi.o benchmark_clock_gettime.c -o benchmark_clock_gettime
	$(CC) $(CFLAGS) computepi.o error.c -o error


.PHONY: clean default

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@ 

check: default
	time ./time_test_baseline
	time ./time_test_openmp_2
	time ./time_test_openmp_4
	time ./time_test_avx
	time ./time_test_avxunroll
	time ./time_test_leibniz
	time ./time_test_leibniz_avx
	time ./time_test_leibniz_avx_unroll
	time ./time_test_montecarlo
	time ./time_test_montecarlo_openmp_2
	time ./time_test_montecarlo_openmp_4

gencsv: default
	for i in `seq 100 5000 100100`; do \
		printf "%d " $$i;\
		./benchmark_clock_gettime $$i; \
	done > result_clock_gettime.csv	

plot: result_clock_gettime.csv
	gnuplot runtime.gp

error: default
	for i in `seq 1000 500 100000`; do \
                printf "%d " $$i;\
                ./error $$i; \
        done > error.csv

ploterror: error.csv
	gnuplot error.gp

clean:
	rm -f $(EXECUTABLE) *.o *.s result_clock_gettime.csv error.csv error.png runtime.png
