#include "InfInt.h"
#include <list>
using namespace std;

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

