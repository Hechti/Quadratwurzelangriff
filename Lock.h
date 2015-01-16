#include "Utilities.h"

#ifndef _LOCK_H_
#define _LOCK_H_

// wird benoetigt um nur ein einziges mal ein Ergebnis in den Speicher zu schreiben
struct Lock 
{
    int *mutex;

    Lock()
    {
        int state = 0;

        CHECK(cudaMalloc((void**) &mutex, sizeof(int)));
        CHECK(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    ~Lock()
    {
        cudaFree(mutex);
    }
    
    __device__ void lock()
    {
        while(atomicCAS(mutex, 0, 1) != 0);
    }
    __device__ void unlock()
    { 
        atomicExch(mutex, 0);
    }
};

#endif