#include "InfInt.h"

#ifndef UTILITIES_H_
#define UTILITIES_H_

typedef struct
{
    InfInt key;
    InfInt data;
} PowData;

void powModulo(const InfInt &basis, const InfInt &exponent, const InfInt &modulus, InfInt &result);
void powInfInt(const InfInt& base, const InfInt& exp, InfInt& result);
void powInfIntMod(const InfInt& base, const InfInt& exp, const InfInt& mod, InfInt& result);

#define CHECK(call)                                                             \
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
