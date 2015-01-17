#include "Utilities.h"

#ifndef _LOCK_H_
#define _LOCK_H_

// wird benoetigt um nur ein einziges mal ein Ergebnis in den Speicher zu schreiben
struct Lock 
{
    int *mutex;
    int *isResultFound;

    Lock()
    {
        int state = 0;
        CHECK(cudaMalloc((void**) &mutex, sizeof(int)));
        CHECK(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));

        int result = 0;
        CHECK(cudaMalloc((void**) &isResultFound, sizeof(int)));
        CHECK(cudaMemcpy(isResultFound, &result, sizeof(int), cudaMemcpyHostToDevice));

    }
    
    ~Lock()
    {
        cudaFree(mutex);
        cudaFree(isResultFound);
    }
    
    __device__ void lock()
    {
        while(atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock()
    { 
        atomicExch(mutex, 0);
    }

    __device__ void foundResult(int &result)
    {
        result = *isResultFound;
    }

    __device__ void setFoundResult(int &result)
    {
        *isResultFound = 1;
        result = *isResultFound;
    }
};

#endif
