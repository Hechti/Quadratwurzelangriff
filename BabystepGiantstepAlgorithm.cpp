#include "InfInt.h"
#include "Utilities.h"
#include "BabystepGiantstepAlgorithm.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>

void babystepGiantstepAlgorithm(const InfInt& n, const InfInt& g, const InfInt& a, std::vector<InfInt>& possibleResults)
{
	InfInt m = (n-1).intSqrt() + 1;
    	printf("\tm: %s\n", m.toString().c_str());
	
	printf("\tTabelle j [");
	std::vector<InfInt> tableGJ;
	for (InfInt j=0; j<m; j++)
	{
		InfInt result;
		powModulo(g, j, n, result);
		tableGJ.push_back(result);
		printf("%s,", result.toString().c_str());
	}
	printf("\b]\n");

	printf("\tTabelle i [");
	std::vector<InfInt> tableGI;
	for (InfInt i=0; i<m-1; i++)
	{
		InfInt exp = (n - 1) - (i * m);
		InfInt tmpErg; 
		powModulo(g, exp, n, tmpErg);
		InfInt result = (a * tmpErg) % n;
		tableGI.push_back(result);
		printf("%s,", result.toString().c_str());
	}
	printf("\b]\n");

	printf("\tPossible Results: [");
	for (InfInt i=0; i<m-1; i++)
	{
		for (InfInt j=0; j<m; j++)
		{
			if (tableGI.at(i.toUnsignedLongLong()) == tableGJ.at(j.toUnsignedLongLong()))
			{
				InfInt result = i * m + j;
				possibleResults.push_back(result);
				printf("%s,", result.toString().c_str());
			}
		}
	}
	printf("\b]\n\n");
}

void printTable(const std::vector<InfInt>& table)
{
	printf("[");
	for (auto value : table)
	{
    		printf("%s,", value.toString().c_str());
	}
	printf("\b]\n");
}
