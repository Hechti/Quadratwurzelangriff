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

#endif
