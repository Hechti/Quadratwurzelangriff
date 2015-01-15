#ifndef BABYSTEP_GIANTSTEP_ALGORITHM_H_
#define BABYSTEP_GIANTSTEP_ALGORITHM_H_

typedef unsigned long long ll;

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &result);
void babystepGiantstepAlgorithmCUDA(const InfInt &n, const InfInt& g, const InfInt& a, InfInt &result);
__global__ void babyStep(const ll *n, const unsigned int *m, const ll *g, ll *mapBabyStep);
__device__ void cudaPow(const ll *basis, const ll *exponent, const ll *modulus, ll *result);
__device__ void getArraySize(const ll *exp, int *arraySize);
__device__ void cudaPowModll(const ll* base, const ll* exp, const ll* mod, ll* result);

#endif
