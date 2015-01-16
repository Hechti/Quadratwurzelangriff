#include "Lock.h"

#ifndef BABYSTEP_GIANTSTEP_ALGORITHM_H_
#define BABYSTEP_GIANTSTEP_ALGORITHM_H_

typedef unsigned long long ll;

typedef struct
{
    unsigned int i;
    unsigned int j;
} CudaResult;

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &result);
void babystepGiantstepAlgorithmCUDA(const InfInt &n, const InfInt& g, const InfInt& a, InfInt &result);
__global__ void giantStep(const ll *n, const unsigned int *m, const ll *g, const ll *a, ll *mapBabyStep, unsigned int *resultI, unsigned int *resultJ, int *foundResult);
__global__ void babyStep(const ll *n, const unsigned int *m, const ll *g, const unsigned int *offset, ll *mapBabyStep); 
__device__ void cudaPow(const ll *basis, const ll *exponent, const ll *modulus, ll *result);
__device__ void getArraySize(const ll *exp, int *arraySize);
__device__ void cudaPowModll(const ll* base, const ll* exp, const ll* mod, ll* result);

// neuer Test
void babyGiant(InfInt &n, InfInt &g, InfInt &a, ll &result);
__global__ void baby(const unsigned int *m, const ll *g, const ll *n, const unsigned int *offset, ll *babyStepTable);
__global__ void giant(const unsigned int *m, const ll *g, const ll *n, const ll *a, const unsigned int *offset, const ll *babyStepTable, CudaResult *result, int *isResultFound, int *mutex, Lock lock);

#endif
