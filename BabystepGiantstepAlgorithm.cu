#include <cuda_runtime.h>
#include "InfInt.h"
#include "Utilities.h"
#include <map>
#include "BabystepGiantstepAlgorithm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void printBabyStepTable(std::map<InfInt, InfInt> mapBabyStep)
{
    
	printf("\tBabyStep j [");
    for (auto &value : mapBabyStep)
    {
        printf("%s,", value.first.toString().c_str());
    }
	printf("\b]\n");
}

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &secretResult)
{
	InfInt m = (n-1).intSqrt() + 1;
    printf("\tm: %s\n", m.toString().c_str());
    
    std::map<InfInt, InfInt> mapBabyStep;
	for (InfInt j=0; j<m; j++)
	{
		InfInt result;
		powModulo(g, j, n, result);
		mapBabyStep[result] = j;
	}

    if (m < InfInt(100))
    {
        printBabyStepTable(mapBabyStep);
    }

	for (InfInt i=0; i<m; i++)
	{
		// InfInt exp = (n - 1) - (i * m);
        InfInt one = 1;
        InfInt exp = (n - m - one) * i;
		InfInt tmpErg; 
		powModulo(g, exp, n, tmpErg);
		InfInt result = (a * tmpErg) % n;
		
        auto it = mapBabyStep.find(result);
        if (it != mapBabyStep.end())
        {
            secretResult = i * m + it->second;
        	printf("\tsecret result: [%s]\n\n", secretResult.toString().c_str());
            return;
        }

	}
}

void babystepGiantstepAlgorithmCUDA(const InfInt &n, const InfInt &g, const InfInt &a, InfInt &result)
{
	InfInt m = (n - 1).intSqrt() + 1;
    
    ll *mapBabyStep = (ll*)malloc(m.toUnsignedLongLong() * sizeof(ll));

    ll *deviceN, *deviceM, *deviceG, *deviceMapBabyStep;
    cudaMalloc((void**) &deviceN, sizeof(ll));
    cudaMalloc((void**) &deviceM, sizeof(ll));
    cudaMalloc((void**) &deviceG, sizeof(ll));
    cudaMalloc((void**) &deviceMapBabyStep, m.toUnsignedLongLong() * sizeof(ll));

    ll value = n.toUnsignedLongLong();
    cudaMemcpy(deviceN, &value, sizeof(ll), cudaMemcpyHostToDevice);
    value = m.toUnsignedLongLong();
    cudaMemcpy(deviceM, &value, sizeof(ll), cudaMemcpyHostToDevice);
    value = g.toUnsignedLongLong();
    cudaMemcpy(deviceG, &value, sizeof(ll), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMapBabyStep, mapBabyStep, m.toUnsignedLongLong() * sizeof(ll), cudaMemcpyHostToDevice);

    babyStep<<<20, 20>>>(deviceN, deviceM, deviceG, deviceMapBabyStep);

    cudaFree(deviceN);
    cudaFree(deviceM);
    cudaFree(deviceG);
    cudaFree(deviceMapBabyStep);
    
    free(mapBabyStep);
}

__global__ void babyStep(const ll *n, const ll *m, const ll *g, ll *mapBabyStep)
{
    ll id = (threadIdx.x + blockIdx.x * blockDim.x);

    if (id < *m)
    {
        printf("id: %d\n", (threadIdx.x + blockIdx.x * blockDim.x));
        ll result;
        cudaPow(g, &id, n, &result);
        printf("pow: %llu\n", result); 
        // mapBabyStep[id]
    }
}

typedef struct
{
    ll key;
    ll data;
} CudaPowData;

__device__ void cudaPow(const ll *basis, const ll *exponent, const ll *modulus, ll *result)
{
    ll check1 = 0;
    ll check2 = 1;

    if (*basis == check1)
    {
        *result = check1;
        return;
    }

    if (*exponent == check1)
    {
        *result = check2;
        return;
    }

    if (*exponent == check2)
    {
        *result = *basis;
        return;
    }

    int arraySize = 0;
    int arrayCount = 0;
    getArraySize(exponent, &arraySize);
    
    CudaPowData *values = new CudaPowData[arraySize];
    
    ll globalExp = 1;
    *result = *basis;

    do
    {
        if ((globalExp + globalExp) <= *exponent)
        {
            *result *= *result;
            *result %= *modulus;
            globalExp *= 2;

            CudaPowData data;
            data.key = globalExp;
            data.data = *result;

            values[arrayCount] = data;
        }
        else
        {
            if ((*exponent - globalExp) == 1)
            {
                *result *= *basis;
                *result %= *modulus;
                globalExp += 1;
            }
            else
            {
                for (int i = arraySize - 1; i >= 0; i--)
                {
                    if ((values[i].key + globalExp) <= *exponent)
                    {
                        *result *= values[i].data;
                        *result %= *modulus;
                        globalExp += values[i].key;
                    }
                }
            }
        }
    } while (globalExp != *exponent);

    delete [] values;
}

__device__ void getArraySize(const ll *exp, int *arraySize)
{
    *arraySize = 0;
    ll value = 1;

    while (value <= *exp)
    {
        value *= 2;
        *arraySize++;
    } 
    *arraySize -= 1;
}
