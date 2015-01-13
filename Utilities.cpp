#include "InfInt.h"
#include <map>
using namespace std;

void powModulo(InfInt *basis, InfInt *exponent, InfInt *modulus, InfInt *result)
{
    InfInt check1 = 0;
    InfInt check2 = 1;

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

    map<InfInt, InfInt> values;

    for (InfInt exp = 1; exp != *exponent; exp *= 2)
    {
        
    }
}

