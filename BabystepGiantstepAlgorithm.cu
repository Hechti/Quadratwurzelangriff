#include <cuda_runtime.h>
#include "Lock.h"
#include "InfInt.h"
#include "Utilities.h"
#include "DiffieHellman.h"
#include "BabystepGiantstepAlgorithm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void babyGiant(InfInt &n, InfInt &g, InfInt &a, InfInt &b, InfInt &result)
{
        const unsigned int MAX_BLOCK_SIZE = 65535;
        const unsigned int MAX_THREAD_SIZE = 1023;
        unsigned int m = ((n-1).intSqrt() + 1).toUnsignedInt();

        unsigned int numberOfBlocks;
        unsigned int numberOfThreads = 1;
        unsigned int offset = 1;

        // Berechnung der Anzahl der benoetigten Threads und einem offset, 
        // da unter umstaenden jeder CUDA-Core mehrere Berechnungen durchfuehren muss
        if (m > MAX_BLOCK_SIZE)
        {
                numberOfBlocks = MAX_BLOCK_SIZE;
                numberOfThreads = (m / MAX_BLOCK_SIZE) + 1;

                if (numberOfThreads >= MAX_THREAD_SIZE)
                {
                        numberOfThreads = MAX_THREAD_SIZE;
                        offset = (m / (MAX_BLOCK_SIZE * MAX_THREAD_SIZE)) + 1;
                }
        }
        else
        {
                numberOfBlocks = m;
        }

        printf("\n\nStartet CUDA with %u blocks, %u threads, offset = %u, and m = %u!\n\n", numberOfBlocks, numberOfThreads, offset, m);

        // Deklaration aller CUDA-Variablen
        ll *hostBabyStepTable; 
        ll *deviceBabyStepTable;
        unsigned int *deviceM;
        ll *deviceN;
        ll *deviceG;
        ll *deviceA;
        ll *deviceB;
        unsigned int *deviceOffset;
        CudaResult hostResultAlice;
        CudaResult *deviceResultAlice;
        CudaResult hostResultBob;
        CudaResult *deviceResultBob;

        // DEBUG
        hostBabyStepTable = new ll[m];

        // Allokiern von Grafikartenspeicher
        CHECK(cudaMalloc((void**) &deviceM, sizeof(unsigned int)));
        CHECK(cudaMalloc((void**) &deviceN, sizeof(ll)));
        CHECK(cudaMalloc((void**) &deviceG, sizeof(ll)));
        CHECK(cudaMalloc((void**) &deviceA, sizeof(ll)));
        CHECK(cudaMalloc((void**) &deviceB, sizeof(ll)));
        CHECK(cudaMalloc((void**) &deviceOffset, sizeof(unsigned int)));
        CHECK(cudaMalloc((void**) &deviceBabyStepTable, m * sizeof(ll)));
        CHECK(cudaMalloc((void**) &deviceResultAlice, sizeof(CudaResult)));
        CHECK(cudaMalloc((void**) &deviceResultBob, sizeof(CudaResult)));

        // Daten auf die Grafikkarte kopieren
        CHECK(cudaMemcpy(deviceM, &m, sizeof(unsigned int), cudaMemcpyHostToDevice));

        ll value = n.toUnsignedLongLong();
        CHECK(cudaMemcpy(deviceN, &value, sizeof(ll), cudaMemcpyHostToDevice));

        value = g.toUnsignedLongLong();
        CHECK(cudaMemcpy(deviceG, &value, sizeof(ll), cudaMemcpyHostToDevice));

        value = a.toUnsignedLongLong();
        CHECK(cudaMemcpy(deviceA, &value, sizeof(ll), cudaMemcpyHostToDevice));

        value = b.toUnsignedLongLong();
        CHECK(cudaMemcpy(deviceB, &value, sizeof(ll), cudaMemcpyHostToDevice));

        CHECK(cudaMemcpy(deviceOffset, &offset, sizeof(unsigned int), cudaMemcpyHostToDevice));

        hostResultAlice.i = 0;
        hostResultAlice.j = 0;
        CHECK(cudaMemcpy(deviceResultAlice, &hostResultAlice, sizeof(CudaResult), cudaMemcpyHostToDevice));

        hostResultBob.i = 0;
        hostResultBob.j = 0;
        CHECK(cudaMemcpy(deviceResultBob, &hostResultBob, sizeof(CudaResult), cudaMemcpyHostToDevice));

        // Fuelle die BabStep Tabelle
        baby<<<numberOfBlocks, numberOfThreads>>>(deviceM, deviceG, deviceN, deviceOffset, deviceBabyStepTable);

        // DEBUG
        CHECK(cudaMemcpy(hostBabyStepTable, deviceBabyStepTable, m * sizeof(ll), cudaMemcpyDeviceToHost));

        if (m < 100)
        {
                printf("Table j: [");
                for (unsigned int i = 0; i < m; i++)
                {
                        printf("%llu,", hostBabyStepTable[i]);
                }
                printf("\b]\n\n");

        }

        // Suche nach Alice's Eingabe
        Lock lockA;
        giant<<<numberOfBlocks, numberOfThreads>>>(deviceM, deviceG, deviceN, deviceA, deviceOffset, deviceBabyStepTable, deviceResultAlice, lockA);

        // Suche nach Bob's Eingabe
        Lock lockB;
        giant<<<numberOfBlocks, numberOfThreads>>>(deviceM, deviceG, deviceN, deviceB, deviceOffset, deviceBabyStepTable, deviceResultBob, lockB);

        // Ausgabe Ergebnis Alice
        CHECK(cudaMemcpy(&hostResultAlice, deviceResultAlice, sizeof(CudaResult), cudaMemcpyDeviceToHost));
        printf("\nAlice:\n");
        printf("i: %u, j: %u\n", hostResultAlice.i, hostResultAlice.j);
        InfInt ergAlice = (InfInt(hostResultAlice.i) * InfInt(m)) + InfInt(hostResultAlice.j);
        printf("Ergebnis: %s\n", ergAlice.toString().c_str());

        // Ausgabe Ergebnis Bob
        CHECK(cudaMemcpy(&hostResultBob, deviceResultBob, sizeof(CudaResult), cudaMemcpyDeviceToHost));
        printf("\nBob:\n");
        printf("i: %u, j: %u\n", hostResultBob.i, hostResultBob.j);
        InfInt ergBob = (InfInt(hostResultBob.i) * InfInt(m)) + InfInt(hostResultBob.j);
        printf("Ergebnis: %s\n", ergBob.toString().c_str());

        InfInt alice(ergAlice);
        InfInt bob(ergBob);
        InfInt pseudo1, pseudo2;
        diffieHellman(n, g, alice, bob, pseudo1, pseudo2, result);

        printf("\n\ncalculated private key: %s\n\n", result.toString().c_str());


        // DEBUG
        delete [] hostBabyStepTable;
        // Grafikkartenspeicher freigeben
        CHECK(cudaFree(deviceM));
        CHECK(cudaFree(deviceN));
        CHECK(cudaFree(deviceG));
        CHECK(cudaFree(deviceA));
        CHECK(cudaFree(deviceB));
        CHECK(cudaFree(deviceOffset));
        CHECK(cudaFree(deviceBabyStepTable));
        CHECK(cudaFree(deviceResultAlice));
        CHECK(cudaFree(deviceResultBob));
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

__global__ void giant(unsigned int *m, ll *g, const ll *n, ll *a, const unsigned int *offset, const ll *babyStepTable, CudaResult *result, Lock lock)
{
        // ID berechnen
        unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int lowerLimit;
        unsigned int higherLimit;

        // create a shared variable and initialize
        __shared__  int isResultFound;
        lock.foundResult(isResultFound);

        // untere und obere Grenze bestimmen
        lowerLimit = id * *offset;
        higherLimit = lowerLimit + *offset;

        // Jede GPU arbeitet ihren Block ab, auszer es wurde ein Ergebnis gefunden
        for (unsigned int i = lowerLimit; i < higherLimit && i < *m && !isResultFound; i++)
        {
                ll exp = *n;
                exp -= *m - 1;
                exp *= exp * i;
                exp %= *n; 

                ll tmpResult = 0;
                cudaPowModll(g, &exp, n, &tmpResult);
                tmpResult *= *a;
                tmpResult %= *n;

                for (unsigned int j = 0; j < *m && !isResultFound; j++)
                {
                        if (tmpResult == babyStepTable[j])
                        {
                                lock.lock();
                                lock.foundResult(isResultFound);

                                if (!isResultFound)
                                {
                                        result->j = j;
                                        result->i = i;
                                        lock.setFoundResult(isResultFound);
                                        isResultFound = true;

                                        printf("found result: (%u, %u) -> %llu\n", i, j, tmpResult);
                                }
                                lock.unlock();

                                return;
                        }
                }
        }
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
                if (((*exp >> i) &1) == 1)
                {
                        break;
                }
        }

        *result = *base;

        for (--i; i >=0; --i)
        {
                *result *= *result;
                *result %= *mod;

                if ((* exp >> i) &1)
                {
                        *result *= *base;
                        *result %= *mod;
                }
        }
}
