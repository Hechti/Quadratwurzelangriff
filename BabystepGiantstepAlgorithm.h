#include "Lock.h"

#ifndef BABYSTEP_GIANTSTEP_ALGORITHM_H_
#define BABYSTEP_GIANTSTEP_ALGORITHM_H_

typedef unsigned long long ll;

typedef struct
{
    unsigned int i;
    unsigned int j;
} CudaResult;

__device__ void cudaPowModll(ll* base, const ll* exp, const ll* mod, ll* result);
void babyGiant(InfInt &n, InfInt &g, InfInt &a, InfInt &b, InfInt &result);
__global__ void baby(const unsigned int *m, const ll *g, const ll *n, const unsigned int *offset, ll *babyStepTable);
__global__ void giant(unsigned int *m, ll *g, const ll *n, ll *a, const unsigned int *offset, const ll *babyStepTable, CudaResult *result, Lock lock);

#endif
