#include <stdio.h>
#include "Utilities.h"
#include "InfInt.h"

int main(void)
{
    InfInt basis = 2;
    InfInt exponent = 8;
    InfInt modulus = 100;
    InfInt result;

    powModulo(basis, exponent, modulus, result);
    printf("Ergebnis: %s\n", result.toString().c_str());

    return 0;
}
