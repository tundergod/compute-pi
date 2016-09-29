#include <stdio.h>
#include "computepi.h"

int main(int argc, char const *argv[])
{
    __attribute__((unused)) int N = 500000000;
    double pi = 0.0;

#if defined(BASELINE)
    pi = compute_pi_baseline(N);
#endif

#if defined(OPENMP_2)
    pi = compute_pi_openmp(N, 2);
#endif

#if defined(OPENMP_4)
    pi = compute_pi_openmp(N, 4);
#endif

#if defined(AVX)
    pi = compute_pi_avx(N);
#endif

#if defined(AVXUNROLL)
    pi = compute_pi_avx_unroll(N);
#endif

#if defined(LB)
    pi = compute_pi_leibniz(N);
#endif

#if defined(LBAVX)
    pi = compute_pi_leibniz_avx(N);
#endif
#if defined(LBAVXUNROLL)
    pi = compute_pi_leibniz_avx_unroll(N);
#endif

#if defined(MC)
    pi = compute_pi_montecarlo(N);
#endif

#if defined(MC_OPENMP_2)
    pi = compute_pi_montecarlo_openmp(N,2);
#endif

#if defined(MC_OPENMP_4)
    pi = compute_pi_montecarlo_openmp(N,4);
#endif
    printf("N = %d , pi = %.25lf\n", N, pi);

    return 0;
}
