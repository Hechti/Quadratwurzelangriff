#include "InfInt.h"
#include "Utilities.h"
#include "DiffieHellman.h"

void diffieHellman(const InfInt& n, const InfInt& g, const InfInt& x, const InfInt& y, InfInt& a, InfInt &b, InfInt &privateKey)
{
	powModulo(g, x, n, a);
	powModulo(g, y, n, b);
    powModulo(a, y, n, privateKey);
}
