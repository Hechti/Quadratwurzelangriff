#include "InfInt.h"
#include <list>
using namespace std;

#include "Utilities.h"

void powModulo(const InfInt &basis, const InfInt &exponent, const InfInt &modulus, InfInt &result)
{
    InfInt check1 = 0;
    InfInt check2 = 1;

    if (basis == check1)
    {
        result = check1;
        return;
    }

    if (&exponent == check1)
    {
        result = check2;
        return;
    }

    if (exponent == check2)
    {
        result = basis;
        return;
    }

    map<InfInt, InfInt> values;
    InfInt currentExp = 1;
    InfInt globalExp = 1;
    result = basis;

    do
    {
        globalExp *= 2;
        
        if ((globalExp - exponent) > 0)
        {
            result *= result;
            currentExp *= 2;

//             values.push_front(result;
        }
        else
        {
            
        }
    } while (globalExp != exponent);
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
