#include <stdio.h>
#include "Utilities.h"
#include "InfInt.h"

void readInput(InfInt &basis, InfInt &exponent, InfInt &modulus, char **argv);
void printInput(InfInt &basis, InfInt &exponent, InfInt &modulus);

int main(int argc, char **argv)
{
    InfInt basis;
    InfInt exponent;
    InfInt modulus;
    InfInt result;

    if (argc == 4)
    {
        readInput(basis, exponent, modulus, argv);
        printInput(basis, exponent, modulus);
    }
    else
    {
        printf("Please use: basis exponent modulus\n");
        return 1;
    }

    powModulo(basis, exponent, modulus, result);
    printf("Ergebnis: %s\n", result.toString().c_str());

    return 0;
}

void readInput(InfInt &basis, InfInt &exponent, InfInt &modulus, char **argv)
{
    basis = argv[1];
    exponent = argv[2];
    modulus = argv[3];
}

void printInput(InfInt &basis, InfInt &exponent, InfInt &modulus)
{
    printf("Basis:    %s\n", basis.toString().c_str());
    printf("Exponent: %s\n", exponent.toString().c_str());
    printf("Modulus:  %s\n", modulus.toString().c_str());
}
