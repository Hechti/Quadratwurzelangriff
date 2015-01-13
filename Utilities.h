#include "InfInt.h"

#ifndef UTILITIES_H_
#define UTILITIES_H_

typedef struct
{
    InfInt key;
    InfInt data;
} PowData;

void powModulo(InfInt &basis, InfInt &exponent, InfInt &modulus, InfInt &result);
void powInfInt(const InfInt& base, const InfInt& exp, InfInt& result);

#endif
