#include <stdio.h>
#include <vector>
#include <time.h>

#include "InfInt.h"
#include "Utilities.h"
#include "DiffieHellman.h"
#include "BabystepGiantstepAlgorithm.h"
#include <chrono>
using namespace std::chrono;

void readInput(InfInt &n, InfInt &g, InfInt &inputAlice, InfInt &inputBob, char **argv);
bool isNumberNegative(const InfInt &number);
bool isInputValide(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob);
void printInput(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob);
void printHelp(void);

int main(int argc, char **argv)
{
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

    auto start11 = high_resolution_clock::now();
    start = clock(); 
    diffieHellman(modulus, basis, inputAlice, inputBob, keyAlice, keyBob, privateKey);
    finish = clock();
    auto finish11 = high_resolution_clock::now();
    long long duration11 = duration_cast<milliseconds>(finish11.time_since_epoch() - start11.time_since_epoch()).count();
    printf("\nAlice sendet: %s\n", keyAlice.toString().c_str());
    printf("Bob sendet:     %s\n", keyBob.toString().c_str());
    printf("private Key:    %s\n", privateKey.toString().c_str());

    // Ausgabe Zeitmessung
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    printf("Duration C++11: %lld ms\n", duration11);

    InfInt possibleKey0, possibleKey1;
    printf("Alice's number:\n\n");
    start11 = high_resolution_clock::now();
    start = clock();
    babystepGiantstepAlgorithm(modulus, basis, keyAlice, possibleKey0);
    finish = clock();
    finish11 = high_resolution_clock::now();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    duration11 = duration_cast<milliseconds>(finish11.time_since_epoch() - start11.time_since_epoch()).count();
    printf("Duration C++11: %lld ms\n", duration11);
    
    printf("Bob's number:\n\n");
    start11 = high_resolution_clock::now();
    start = clock();
    babystepGiantstepAlgorithm(modulus, basis, keyBob, possibleKey1);
    finish = clock();
    finish11 = high_resolution_clock::now();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%.4f duration\n", duration);
    duration11 = duration_cast<milliseconds>(finish11.time_since_epoch() - start11.time_since_epoch()).count();
    printf("Duration C++11: %lld ms\n", duration11);
    
    diffieHellman(modulus, basis, possibleKey0, possibleKey1, keyAlice, keyBob, privateKey);
    printf("private Key:    %s\n", privateKey.toString().c_str());

    start11 = high_resolution_clock::now();
    start = clock();
    ll erg;
    babyGiant(modulus, basis, keyBob, erg);    
    // babystepGiantstepAlgorithmCUDA(modulus, basis, keyBob, possibleKey1);
    finish = clock();
    finish11 = high_resolution_clock::now();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f duration\n", duration);
    duration11 = duration_cast<milliseconds>(finish11.time_since_epoch() - start11.time_since_epoch()).count();
    printf("Duration C++11: %lld ms\n", duration11);

    return 0;
}

void readInput(InfInt &n, InfInt &g, InfInt &inputAlice, InfInt &inputBob, char **argv)
{
    n = argv[1];
    g = argv[2];
    inputAlice = argv[3];
    inputBob = argv[4];

    if (!isInputValide(n, n, inputAlice, inputBob))
    {
        printHelp();
        exit(1);
    }
}

bool isInputValide(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob)
{
    return !isNumberNegative(n) && !isNumberNegative(g) && !isNumberNegative(inputAlice) && !isNumberNegative(inputBob);
}

bool isNumberNegative(const InfInt &number)
{
    InfInt zero = 0;

    return number < zero;
}

void printInput(const InfInt &n, const InfInt &g, const InfInt &inputAlice, const InfInt &inputBob)
{
    printf("n:      %s\n", n.toString().c_str());
    printf("g:      %s\n", g.toString().c_str());
    printf("Alice:  %s\n", inputAlice.toString().c_str());
    printf("Bob:    %s\n", inputBob.toString().c_str());
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
