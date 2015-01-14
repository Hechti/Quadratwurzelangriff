#include <stdio.h>
#include <vector>
#include <time.h>

#include "InfInt.h"
#include "Utilities.h"
#include "DiffieHellman.h"
#include "BabystepGiantstepAlgorithm.h"


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
    
    InfInt n = 29;
    InfInt g = 11;
    InfInt x = 28;
    InfInt y = 17;

    InfInt a;
    InfInt b;
    diffieHellman(n, g, x, y, a, b);
    printf("\nAlice sendet: %s\n", a.toString().c_str());
    printf("Bob sendet: %s\n", b.toString().c_str());

    std::vector<InfInt> possibleKeys;
    printf("Alice's number:\n\n");
    babystepGiantstepAlgorithm(n, g, a, possibleKeys);
    printf("Bob's number:\n\n");
    babystepGiantstepAlgorithm(n, g, b, possibleKeys);

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
