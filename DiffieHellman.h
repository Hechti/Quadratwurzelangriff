#include "InfInt.h"

#ifndef DIFFIE_HELLMAN_H_
#define DIFFIE_HELLMAN_H_

void diffieHellman(const InfInt& n, const InfInt& g, const InfInt& x, const InfInt& y, InfInt& a, InfInt &b, InfInt &privateKey);

#endif
