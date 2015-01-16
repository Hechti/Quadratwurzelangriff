#include "Utilities.h"

#ifndef _LOCK_H_
#define _LOCK_H_

// wird benoetigt um nur ein einziges mal ein Ergebnis in den Speicher zu schreiben
struct Lock 
{
    int *mutex;
    bool *foundResult;

    Lock()
    {
        int state = 0;
        bool result = false;

        CHECK(cudaMalloc((void**) &mutex, sizeof(int)));
        CHECK(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));

        CHECK(cudaMalloc((void**) &foundResult, sizeof(bool)));
        CHECK(cudaMemcpy(foundResult, &result, sizeof(bool), cudaMemcpyHostToDevice));
    }
    
    ~Lock()
    {
        cudaFree(mutex);
        cudaFree(foundResult);
    }
    
    __device__ void lock()
    {
        while(atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock()
    { 
        atomicExch(mutex, 0);
    }

    __device__ void isResultFound(bool *result)
    {
        *result = *foundResult;
    }

    __device__ void setResultFound(bool *result)
    {
        *result = *foundResult = true;
    }
};

#endif
