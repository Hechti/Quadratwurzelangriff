#include <stdio.h>
#include "Utilities.h"
#include "InfInt.h"

#include <time.h>

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

    clock_t start, finish;
    double duration;

    start = clock();
    powModulo(basis, exponent, modulus, result);
    finish = clock();
    printf("Ergebnis: %s\n", result.toString().c_str());
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f duration\n", duration);

    start = clock();
    powInfIntMod(basis, exponent, modulus, result);
    finish = clock();
    printf("Ergebnis: %s\n", result.toString().c_str());
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f duration\n", duration);

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
