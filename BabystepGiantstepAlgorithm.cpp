#include "InfInt.h"
#include "Utilities.h"
#include "BabystepGiantstepAlgorithm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <map>

void printBabyStepTable(std::map<InfInt, InfInt> mapBabyStep)
{
    
	printf("\tBabyStep j [");
    for (auto &value : mapBabyStep)
    {
        printf("%s,", value.first.toString().c_str());
    }
	printf("\b]\n");
}

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, InfInt &secretResult)
{
	InfInt m = (n-1).intSqrt() + 1;
    printf("\tm: %s\n", m.toString().c_str());
    
    std::map<InfInt, InfInt> mapBabyStep;
	for (InfInt j=0; j<m; j++)
	{
		InfInt result;
		powModulo(g, j, n, result);
		mapBabyStep[result] = j;
	}

    if (m < InfInt(100))
    {
        printBabyStepTable(mapBabyStep);
    }

	for (InfInt i=0; i<m; i++)
	{
		// InfInt exp = (n - 1) - (i * m);
        InfInt one = 1;
        InfInt exp = (n - m - one) * i;
		InfInt tmpErg; 
		powModulo(g, exp, n, tmpErg);
		InfInt result = (a * tmpErg) % n;
		
        auto it = mapBabyStep.find(result);
        if (it != mapBabyStep.end())
        {
            secretResult = i * m + it->second;
        	printf("\tsecret result: [%s]\n\n", secretResult.toString().c_str());
            return;
        }

	}
}
