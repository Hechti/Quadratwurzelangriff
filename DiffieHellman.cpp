#include "cump.h"
#include "gmp.h"
#include "DiffieHellman.h"

void diffieHellman(const mpz_t n, const mpz_t g, const mpz_t x, const mpz_t y, mpz_t a, mpz_t b, mpz_t privateKey)
{
    mpz_powm(a, g, x, n); 
    mpz_powm(b, g, y, n); 
    mpz_powm(privateKey, a, y, n); 
}
