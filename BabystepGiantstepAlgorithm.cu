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

__global__ void baby(const unsigned int *m, const ll *g, const ll *n, const unsigned int *offset, ll *babyStepTable)
{
    // ID berechnen
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lowerLimit;
    unsigned int higherLimit;

    // untere und obere Grenze bestimmen
    lowerLimit = id * *offset;
    higherLimit = lowerLimit + *offset;

    // J-Tabelle berechnen mit: g^j mod n
    for (unsigned int j = lowerLimit; j < higherLimit && j < *m; j++)
    {
        cudaPowModll(g, (ll*) &j, n, &babyStepTable[j]);
    }
}

typedef struct
{
    unsigned int i;
    unsigned int j;
} CudaResult;

__global__ void giant(const unsigned int *m, const ll *g, const ll *n, const ll *a, const unsigned int *offset, const ll *babyStepTable, CudaResult *result, int *isResultFound)
{
    // ID berechnen
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lowerLimit;
    unsigned int higherLimit;

    // untere und obere Grenze bestimmen
    lowerLimit = id * *offset;
    higherLimit = lowerLimit + *offset;

    // Jede GPU arbeitet ihren Block ab, auszer es wurde ein Ergebnis gefunden
    for (unsigned int i = lowerLimit; i < higherLimit && i < *m && !*isResultFound; i++)
    {
        ll exp = *m;
        exp -= *n;
        exp = (exp -1) * i;
        
        ll tmpResult = 0;
        cudaPowModll(g, &exp, n, &tmpResult);
        tmpResult *= *a;

        for (unsigned int j = 0; j < *m && !*isResultFound; j++)
        {
            if (tmpResult == babyStepTable[j])
            {
                // Atomares zuweisen notwendig, da es vorkommen kann, 
                // dass mehrere gueltige Ergebnisse gefunden werden
                atomicAdd(isResultFound, 1);
                atomicAdd(&(result->j), babyStepTable[j]);
                atomicAdd(&(result->i), i);

                return;
            }
        }
    }
}

void babyGiant(const InfInt& n, const InfInt& g, const InfInt& a, ll &result)
{
	const unsigned int MAX_BLOCK_SIZE = 65536;
    const unsigned int MAX_THREAD_SIZE = 1024;
    unsigned int m = ((n-1).intSqrt() + 1).toUnsignedInt();
    
    unsigned int numberOfBlocks;
    unsigned int numberOfThreads = 1;
    unsigned int offset = 0;

    // Berechnung der Anzahl der benoetigten Threads und einem offset, 
    // da unter umstaenden jeder CUDA-Core mehrere Berechnungen durchfuehren muss
    if (m >= 65536)
    {
        numberOfBlocks = MAX_BLOCK_SIZE;
        numberOfThreads = (m / MAX_BLOCK_SIZE) + 1;

        if (numberOfThreads >= 1024)
        {
            offset = (m / (MAX_BLOCK_SIZE * MAX_THREAD_SIZE)) + 1;
        }
    }
    else
    {
        numberOfBlocks = m;
    }

    // Deklaration aller CUDA-Variablen
    // ll *hostBabyStepTable; 
    ll *deviceBabyStepTable;
    unsigned int *deviceM;
    ll *deviceN;
    ll *deviceG;
    ll *deviceA;
    unsigned int *deviceOffset;
    CudaResult *deviceResult;
    int isResultFound = 0;
    int *deviceIsResultFound;

    // Allokiern von Grafikartenspeicher
    CHECK(cudaMalloc((void**) &deviceM, sizeof(unsigned int)));
    CHECK(cudaMalloc((void**) &deviceN, sizeof(ll)));
    CHECK(cudaMalloc((void**) &deviceG, sizeof(ll)));
    CHECK(cudaMalloc((void**) &deviceA, sizeof(ll)));
    CHECK(cudaMalloc((void**) &deviceOffset, sizeof(unsigned int)));
    CHECK(cudaMalloc((void**) &deviceBabyStepTable, m * sizeof(ll)));
    CHECK(cudaMalloc((void**) &deviceResult, sizeof(CudaResult)));
    CHECK(cudaMalloc((void**) &deviceIsResultFound, sizeof(int)));

    // Daten auf die Grafikkarte kopieren
    CHECK(cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice));
    ll value = n.toUnsignedLongLong();
    CHECK(cudaMemcpy(deviceN, &value, sizeof(ll), cudaMemcpyHostToDevice));
    value = g.toUnsignedLongLong();
    CHECK(cudaMemcpy(deviceG, &value, sizeof(ll), cudaMemcpyHostToDevice));
    value = a.toUnsignedLongLong();
    CHECK(cudaMemcpy(deviceA, &value, sizeof(ll), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceOffset, &offset, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(deviceIsResultFound, &isResultFound, sizeof(int), cudaMemcpyHostToDevice));

    baby<<<numberOfBlocks, numberOfThreads>>>(deviceM, deviceG, deviceN, deviceOffset, deviceBabyStepTable);
    giant<<<numberOfBlocks, numberOfThreads>>>(deviceM, deviceG, deviceN, deviceA, deviceOffset, deviceBabyStepTable, deviceResult, deviceIsResultFound);

    CudaResult hostResult;
    CHECK(cudaMemcpy(&hostResult, deviceResult, sizeof(CudaResult), cudaMemcpyDeviceToHost));

    printf("i: %u, j: %u\n", hostResult.i, hostResult.j);

    // Grafikkartenspeicher freigeben
    CHECK(cudaFree(deviceM));
    CHECK(cudaFree(deviceN));
    CHECK(cudaFree(deviceG));
    CHECK(cudaFree(deviceA));
    CHECK(cudaFree(deviceOffset));
    CHECK(cudaFree(deviceBabyStepTable));
    CHECK(cudaFree(deviceResult));
    CHECK(cudaFree(deviceIsResultFound));
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
    unsigned int *deviceM, *deviceOffset;
    
    cudaMalloc((void**) &deviceN, sizeof(ll));
    cudaMalloc((void**) &deviceM, sizeof(unsigned int));
    cudaMalloc((void**) &deviceG, sizeof(ll));
    cudaMalloc((void**) &deviceOffset, sizeof(unsigned int));

    ll value = n.toUnsignedLongLong();
    cudaMemcpy(deviceN, &value, sizeof(ll), cudaMemcpyHostToDevice);
    value = g.toUnsignedLongLong();
    cudaMemcpy(deviceG, &value, sizeof(ll), cudaMemcpyHostToDevice);
    // cudaMemcpy(deviceMapBabyStep, mapBabyStep, m * sizeof(ll), cudaMemcpyHostToDevice);
   
    if (babyTableRowSize == 1)
    {
        value = 0;
        cudaMemcpy(deviceOffset, &value, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &deviceMapBabyStep, m * sizeof(ll));
        
        babyStep<<<m, 1>>>(deviceN, deviceM, deviceG, deviceOffset, deviceMapBabyStep);
        cudaMemcpy(babyTable[0], deviceMapBabyStep, m * sizeof(ll), cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMalloc((void**) &deviceMapBabyStep, BABY_TABLE_COLOUMN_SIZE * sizeof(ll));
        cudaMemcpy(deviceM, &BABY_TABLE_COLOUMN_SIZE, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        for (unsigned int i = 0; i < babyTableRowSize - 1; i++)
        {
            cudaMemcpy(deviceOffset, &i, sizeof(unsigned int), cudaMemcpyHostToDevice);
            babyStep<<<BABY_TABLE_COLOUMN_SIZE, 1>>>(deviceN, deviceM, deviceG, deviceOffset, deviceMapBabyStep);
            cudaMemcpy(babyTable[i], deviceMapBabyStep, BABY_TABLE_COLOUMN_SIZE * sizeof(ll), cudaMemcpyDeviceToHost);
        }
        
        cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice);
        unsigned int offset = babyTableRowSize - 1;
        cudaMemcpy(deviceOffset, &offset, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        babyStep<<<m, 1>>>(deviceN, deviceM, deviceG, deviceOffset, deviceMapBabyStep);
        cudaMemcpy(babyTable[babyTableRowSize - 1], deviceMapBabyStep, m * sizeof(ll), cudaMemcpyDeviceToHost);
    }

    if (m <= 100)
    {
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
    }

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

__global__ void babyStep(const ll *n, const unsigned int *m, const ll *g, const unsigned int *offset, ll *mapBabyStep) 
{
	const unsigned int BABY_TABLE_COLOUMN_SIZE = 65536;
    ll id = blockIdx.x + (BABY_TABLE_COLOUMN_SIZE * *offset);
    cudaPowModll(g, &id, n, &mapBabyStep[id]);
}

__global__ void giantStep(const ll *n, const unsigned int *m, const ll *g, const ll *a, ll *mapBabyStep, unsigned int *resultI, unsigned int *resultJ, int *foundResult)
{
    if (!foundResult)
    {
        ll id = blockIdx.x;
        ll localN, localM;
        localN = *n;
        localM = *m;
        ll exp = (localN - localM - 1) * id;
        ll powResult;
        cudaPowModll(g, &exp, n, &powResult);
        powResult = (powResult * *a) % *n;

        for (unsigned int i = 0; i < *m; i++)
        {
                if (mapBabyStep[i] == powResult && !foundResult)
                {
                    atomicAdd(foundResult, 1);
                    atomicAdd(resultJ, i);
                    atomicAdd(resultI, id);
                    return;
                }
        }
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
