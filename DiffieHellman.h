#include "cump.h"
#include "gmp.h"

#ifndef DIFFIE_HELLMAN_H_
#define DIFFIE_HELLMAN_H_

void diffieHellman(const mpz_t n, const mpz_t g, const mpz_t x, const mpz_t y, mpz_t a, mpz_t b, mpz_t privateKey);

#endif
