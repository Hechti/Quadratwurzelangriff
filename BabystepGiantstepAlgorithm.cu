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
	const unsigned int BABY_TABLE_COLOUMN_SIZE = 65536;
    unsigned int m = ((n - 1).intSqrt() + 1).toUnsignedInt();
    
    printf("m: %u\n", m);

    unsigned int babyTableRowSize;
    ll **babyTable;
    if (m < BABY_TABLE_COLOUMN_SIZE)
    {
        babyTable = new ll*[1];
        babyTable[0] = new ll[m];
        
        babyTableRowSize = 1;
    }
    else
    {
        babyTableRowSize = m / BABY_TABLE_COLOUMN_SIZE;
        babyTableRowSize += 1;

        babyTable = new ll*[babyTableRowSize];

        for (int i = 0; i < babyTableRowSize; i++)
        {
            babyTable[i] = new ll[BABY_TABLE_COLOUMN_SIZE];
        }   
    }



    ll *mapBabyStep = (ll*)malloc(m * sizeof(ll));
    ll *deviceN, *deviceG, *deviceMapBabyStep;
    unsigned int *deviceM;
    
    cudaMalloc((void**) &deviceN, sizeof(ll));
    cudaMalloc((void**) &deviceM, sizeof(unsigned int));
    cudaMalloc((void**) &deviceG, sizeof(ll));

    ll value = n.toUnsignedLongLong();
    cudaMemcpy(deviceN, &value, sizeof(ll), cudaMemcpyHostToDevice);
    value = g.toUnsignedLongLong();
    cudaMemcpy(deviceG, &value, sizeof(ll), cudaMemcpyHostToDevice);
    // cudaMemcpy(deviceMapBabyStep, mapBabyStep, m * sizeof(ll), cudaMemcpyHostToDevice);
    
    if (babyTableRowSize == 1)
    {
        cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &deviceMapBabyStep, m * sizeof(ll));
        babyStep<<<m, 1>>>(deviceN, deviceM, deviceG, deviceMapBabyStep);
        cudaMemcpy(babyTable[0], deviceMapBabyStep, m * sizeof(ll), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMalloc((void**) &deviceMapBabyStep, BABY_TABLE_COLOUMN_SIZE * sizeof(ll));
        cudaMemcpy(deviceM, &BABY_TABLE_COLOUMN_SIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < babyTableRowSize - 1; i++)
        {
            babyStep<<<BABY_TABLE_COLOUMN_SIZE, 1>>>(deviceN, deviceM, deviceG, deviceMapBabyStep);
            cudaMemcpy(babyTable[i], deviceMapBabyStep, BABY_TABLE_COLOUMN_SIZE * sizeof(ll), cudaMemcpyDeviceToHost);
        }
        
        cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice);
        babyStep<<<m, 1>>>(deviceN, deviceM, deviceG, deviceMapBabyStep);
        cudaMemcpy(babyTable[babyTableRowSize - 1], deviceMapBabyStep, m * sizeof(ll), cudaMemcpyDeviceToHost);
    }

    printf("[");
    for (int i = 0; i < babyTableRowSize - 1; i++)
    {
        for (int j = 0; j < BABY_TABLE_COLOUMN_SIZE - 1; j++)
        {
            printf("%llu,", babyTable[i][j]);
        }
    }
    for (int j = 0; j < m; j++)
    {
        printf("%llu,", babyTable[babyTableRowSize - 1][j]);
    }
    printf("\b]\n\n");

    cudaFree(deviceN);
    cudaFree(deviceM);
    cudaFree(deviceG);
    cudaFree(deviceMapBabyStep);
    
    for (int i = 0; i < babyTableRowSize; i++) 
    {
        delete [] babyTable[i];
    }

    delete [] babyTable;

    free(mapBabyStep);
}

__global__ void babyStep(const ll *n, const unsigned int *m, const ll *g, ll *mapBabyStep)
{
    ll id = blockIdx.x;//(threadIdx.x + blockIdx.x * blockDim.x);
    /*ll *maxSize = new ll[1073741823];

    for (int i = 0; i < 1073741823; i++)
    {
        maxSize[i] = i;
    }*/

    // printf("ID: %llu, n: %llu, m: %u, g: %llu\n", id, *n, *m, *g);

    //if (id < *m)
    //{
        // printf("id: %d\n", blockIdx.x);//(threadIdx.x + blockIdx.x * blockDim.x));
        // ll result;
        cudaPowModll(g, &id, n, &mapBabyStep[id]);
        
        printf("pow: %llu\n", mapBabyStep[id]); 
       // printf("RAM SIZE: %llu\n", maxSize[242474]); 
       // mapBabyStep[id] = result;
    //}

    //delete [] maxSize;
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

__device__ void cudaPowModll(const ll* base, const ll* exp, const ll* mod, ll* result)
{
	if (*exp == 0)
	{
		*result = 1;
		return;
	}

	int i;
	for (i = 62; i>=1; --i)
	{
		if (((*exp>>i)&1) == 1)
		{
			break;
		}
	}
	*result = *base;
	for (--i; i >=0; --i)
	{
		*result *= *result;
		*result %= *mod;
		if ((*exp>>i)&1)
		{
			*result *= *base;
			*result %= *mod;
		}
	}
}
