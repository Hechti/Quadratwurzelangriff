#include <map>
using namespace std;

#include "Utilities.h"

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

void powInfInt(const InfInt& base, const InfInt& exp, InfInt& result)
{
	unsigned long long expll = exp.toUnsignedLongLong();

	int i;
	for (i = 62; i>=1; --i)
	{
		if (((expll>>i)&1) == 1)
		{
			break;
		}
	}
	result = base;
	for (--i; i >=0; --i)
	{
		result = result * result;
		if ((expll>>i)&1)
		{
			result = result * base;
		}
	}
}
