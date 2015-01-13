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

    if (exponent == check1)
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
    // InfInt currentExp = 1;
    InfInt globalExp = 1;
    result = basis;

    do
    {
        if ((globalExp + globalExp) <= exponent)
        {
            result *= result;
            result %= modulus;

            // currentExp *= 2;
            globalExp *= 2;

            PowData data;
            data.key = globalExp;
            data.data = result;

            values.push_front(data);
        }
        else
        {
            // printf("exp: %s, result: %s\n", globalExp.toString().c_str(), result.toString().c_str());
            // InfInt tt = globalExp - exponent;
            // printf("gExp - exp = %s\n", tt.toString().c_str());
            // printf("gExp = %s\nexp =  %s\n", globalExp.toString().c_str(), exponent.toString().c_str());
            
            if ((exponent - globalExp) == 1)
            {
                result *= basis;
                result %= modulus;
                globalExp += 1;

                // printf("gExp - exp = 1)\n");
            }
            else
            {
                for (auto data : values)
                {
                    if ((data.key + globalExp) <= exponent)
                    {
                        // printf("data found\n");
                        result *= data.data;
                        result %= modulus;
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

void powInfIntMod(const InfInt& base, const InfInt& exp, const InfInt& mod, InfInt& result)
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
		result *= result;
		result %= mod;
		if ((expll>>i)&1)
		{
			result *= base;
                        result %= mod;
		}
	}
}
