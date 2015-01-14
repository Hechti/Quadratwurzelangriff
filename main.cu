#include <stdio.h>
#include <vector>
#include <time.h>
#include "BabystepGiantstepAlgorithm.h"
#include "DiffieHellman.h"
#include "BabystepGiantstepAlgorithm.h"
#include "cump.h"
#include "gmp.h"

void readInput(mpz_t n, mpz_t g, mpz_t inputAlice, mpz_t inputBob, char **argv);
void createNumber(mpz_t number, const char *value);
bool isNumberNegative(const mpz_t number);
bool isInputValide(const mpz_t n, const mpz_t g, const mpz_t inputAlice, const mpz_t inputBob);
void printInput(const mpz_t n, const mpz_t g, const mpz_t inputAlice, const mpz_t inputBob);
void printHelp(void);
void print(const char *message, const mpz_t value);

int main(int argc, char **argv)
{
    mpz_t basis;
    mpz_t modulus;
    mpz_t inputAlice;
    mpz_t inputBob;
    mpz_t keyAlice;
    mpz_t keyBob;
    mpz_t privateKey;

    if (argc == 5)
    {
        readInput(modulus, basis, inputAlice, inputBob, argv);
        printInput(modulus, basis, inputAlice, inputBob);
    }
    else
    {
        printHelp();
        return 1;
    }
    
    clock_t start, finish;
    double duration;

    start = clock();
    diffieHellman(modulus, basis, inputAlice, inputBob, keyAlice, keyBob, privateKey);
    print("\nAlice sendet: ", keyAlice);
    print("Bob sendet:     ", keyBob);
    print("private Key:    ", privateKey);
    
    mpz_t possibleAliceInput;
    mpz_t possibleBobInput;
    mpz_init(possibleAliceInput);
    mpz_init(possibleBobInput);

    printf("Alice's number:\n\n");
    babystepGiantstepAlgorithm(modulus, basis, keyAlice, possibleAliceInput);
    
    printf("Bob's number:\n\n");
    babystepGiantstepAlgorithm(modulus, basis, keyBob, possibleBobInput);
    
    diffieHellman(modulus, basis, possibleAliceInput, possibleBobInput, keyAlice, keyBob, privateKey);
    print("calculatet private Key: ", privateKey);
    
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    
    return 0;
}

void readInput(mpz_t n, mpz_t g, mpz_t inputAlice, mpz_t inputBob, char **argv)
{
    createNumber(n, argv[1]);
    createNumber(g, argv[2]);
    createNumber(inputAlice, argv[3]);
    createNumber(inputBob, argv[4]);
}

void createNumber(mpz_t number, const char *value)
{
    mpz_init(number);
    mpz_set_str(number, value, 10);
}

bool isInputValide(const mpz_t n, const mpz_t g, const mpz_t inputAlice, const mpz_t inputBob)
{
    return isNumberNegative(n) && isNumberNegative(g) && isNumberNegative(inputAlice) && isNumberNegative(inputBob);
}

bool isNumberNegative(const mpz_t number)
{
    return mpz_sgn(number) < 0;
}

void printInput(const mpz_t n, const mpz_t g, const mpz_t inputAlice, const mpz_t inputBob)
{
    print("n:     ", n);
    print("g:     ", g);
    print("Alice: ", inputAlice);
    print("Bob:  ", inputBob);
}

void printHelp(void)
{
    printf("Please use: quadratwurzelangriff n g x y\n");
    printf("n, g, x, y >= 0\n");
    printf("n = prime number and (n - 1) / 2 si prime too\n");
    printf("g = is prime root of n\n");
    printf("x = secret input from Alice\n");
    printf("y = secret input from Bob\n");
}

void print(const char *message, const mpz_t value)
{
    printf("%s", message);
    mpz_out_str(stdout, 10, value);
    printf("\n");
}
