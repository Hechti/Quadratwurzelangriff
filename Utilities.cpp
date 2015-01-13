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

    list<PowData> values;
    InfInt currentExp = 1;
    InfInt globalExp = 1;
    result = basis;

    do
    {
        if ((globalExp + globalExp) < exponent)
        {
            result *= result;
            
            currentExp *= 2;
            globalExp *= 2;

            PowData data;
            data.key = currentExp;
            data.data = result;

            values.push_front(data);
        }
        else
        {
            if ((globalExp - exponent) == 1)
            {
                result *= basis;
                globalExp += 1;
            }
            else
            {
                for (auto data : values)
                {
                    if ((data.key + currentExp) <= exponent)
                    {
                        result *= data.data;
                        globalExp += data.key;
                    }
                }
            }
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
