#ifndef _DEVICEUTILITIES_H
#define _DEVICEUTILITIES_H

#include <cuda_runtime.h>

typedef unsigned long long ll;

__device__ ll device_mulmod(ll a, ll b, ll m);
__device__ ll device_modpow(ll base, ll exp, ll modulus);

#define CHECK(call)                                                     \
{                                                                       \
    const cudaError_t error = call;                                     \
    if (error != cudaSuccess)                                           \
    {                                                                   \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);          \
        fprintf(stderr, "code: %d, reason: %s\n", error,                \
        cudaGetErrorString(error));                                     \
    }                                                                   \
}

#endif
