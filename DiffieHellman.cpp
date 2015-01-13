#include "InfInt.h"
#include "Utilities.h"
#include "DiffieHellman.h"

void diffieHellman(const InfInt& n, const InfInt& g, const InfInt& x, const InfInt& y, InfInt& a, InfInt &b)
{
	powModulo(g, x, n, a);
	powModulo(g, y, n, b);
}
