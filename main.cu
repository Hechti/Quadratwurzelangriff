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
    
    /*
    InfInt basis;
    InfInt modulus;
    InfInt inputAlice;
    InfInt inputBob;
    InfInt keyAlice;
    InfInt keyBob;
    InfInt privateKey;

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
    powModulo(basis, inputAlice, modulus, keyAlice);
    finish = clock();
    printf("Ergebnis: %s\n", keyAlice.toString().c_str());
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f duration\n", duration);

    start = clock();
    finish = clock();
    printf("Ergebnis: %s\n", keyAlice.toString().c_str());
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f duration\n", duration);
    
    diffieHellman(modulus, basis, inputAlice, inputBob, keyAlice, keyBob, privateKey);
    printf("\nAlice sendet: %s\n", keyAlice.toString().c_str());
    printf("Bob sendet:     %s\n", keyBob.toString().c_str());
    printf("private Key:    %s\n", privateKey.toString().c_str());
    
    InfInt possibleKey0, possibleKey1;
    printf("Alice's number:\n\n");
    start = clock();
    babystepGiantstepAlgorithm(modulus, basis, keyAlice, possibleKey0);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    
    printf("Bob's number:\n\n");
    start = clock();
    babystepGiantstepAlgorithm(modulus, basis, keyBob, possibleKey1);
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    
    diffieHellman(modulus, basis, possibleKey0, possibleKey1, keyAlice, keyBob, privateKey);
    printf("private Key:    %s\n", privateKey.toString().c_str());

    babystepGiantstepAlgorithmCUDA(modulus, basis, keyBob, possibleKey1);
    */
    return 0;
}

void print(const char *message, const mpz_t value)
{
    printf("%s", message);
    mpz_out_str(stdout, 10, value);
    printf("\n");
}

// void readInput(InfInt &n, InfInt &g, InfInt &inputAlice, InfInt &inputBob, char **argv)
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

// bool isInputValide(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob)
bool isInputValide(const mpz_t n, const mpz_t g, const mpz_t inputAlice, const mpz_t inputBob)
{
    return isNumberNegative(n) && isNumberNegative(g) && isNumberNegative(inputAlice) && isNumberNegative(inputBob);
}

bool isNumberNegative(const mpz_t number)
{
    return mpz_sgn(number) < 0;
}

// void printInput(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob)
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
