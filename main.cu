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
    InfInt erg;

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
    
    // Berechnung des private keys mit Dieffie Hellman
    diffieHellman(modulus, basis, inputAlice, inputBob, keyAlice, keyBob, privateKey);
    printf("\nAlice sendet: %s\n", keyAlice.toString().c_str());
    printf("Bob sendet:     %s\n", keyBob.toString().c_str());
    printf("private Key:    %s\n", privateKey.toString().c_str());

    // Berechnung des private keys mit CUDA 
    auto start = high_resolution_clock::now();
    babyGiant(modulus, basis, keyAlice, keyBob, erg);    
    auto finish = high_resolution_clock::now();
   
    // Ausgabe Zeitmessung
    auto duration = duration_cast<milliseconds>(finish.time_since_epoch() - start.time_since_epoch()).count();
    printf("Duration: %ldms\n", duration);

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
