#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "computepi.h"
int main(int argc, char const *argv[])
{
    if(argc < 2) return -1;
    int N = atoi(argv[1]);

    double pi;

    //baseline
    pi = compute_pi_baseline(N);
    printf("%.15lf ",pi - M_PI);

    //leibniz
    pi = compute_pi_leibniz(N);
    printf("%.15lf ",pi - M_PI);

    //monte carlo
    pi = compute_pi_montecarlo(N);
    printf("%.15lf 0\n",pi - M_PI);

    return 0;
}
